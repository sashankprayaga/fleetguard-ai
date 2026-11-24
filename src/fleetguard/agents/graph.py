from __future__ import annotations

import json
import os
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from fleetguard.agents.tools import (
    calculate_dispatch_roi,
    list_top_flagged_vehicles,
    query_vehicle_telemetry_and_shap,
    retrieve_oem_repair_sop,
)
from fleetguard.mlops.tracker import configure_mlflow

load_dotenv()


# =============================================================================
# 1. SHARED GRAPH STATE (The "clipboard" passed from node to node)
# =============================================================================
class TriageAgentState(TypedDict):
    user_query: str
    vehicle_id: str
    telemetry_context: dict
    sop_context: dict
    roi_context: dict
    route_taken: str
    llm_backend: str
    final_work_order: str


def _get_ollama_chat_model():
    """Initialize local ChatOllama using OLLAMA_MODEL from .env (default: llama3.2)."""
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b").strip()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    if not ollama_model:
        return None, "Offline Deterministic Synthesizer"

    from langchain_ollama import ChatOllama

    return ChatOllama(model=ollama_model, base_url=base_url, temperature=0.1), f"Ollama ({ollama_model})"


# =============================================================================
# 2. GRAPH NODES (Each takes `state` and returns a partial dict update)
# =============================================================================
def node_gather_vehicle_context(state: TriageAgentState) -> dict:
    """Step 1: Look up the truck's latest telemetry & SHAP drivers in SQLite."""
    vid = state.get("vehicle_id", "").strip()
    if not vid:
        top_list = json.loads(list_top_flagged_vehicles.invoke({"limit": 1}))
        vid = top_list[0]["vehicle_id"] if top_list else "TRK-1001"

    raw_json = query_vehicle_telemetry_and_shap.invoke({"vehicle_id": vid})
    telemetry = json.loads(raw_json)
    return {"vehicle_id": vid, "telemetry_context": telemetry}


def route_by_risk_tier(state: TriageAgentState) -> Literal["urgent_triage", "routine_clearance"]:
    """Conditional Router: Branch high-risk trucks to RAG+LLM; fast-clear healthy ones."""
    telemetry = state.get("telemetry_context", {})
    if "error" in telemetry:
        return "routine_clearance"

    prob = float(telemetry.get("failure_prob_72h", 0.0))
    return "urgent_triage" if prob >= 0.67 else "routine_clearance"


def node_routine_clearance(state: TriageAgentState) -> dict:
    """Fast-path node when 72h breakdown risk is below the 67% alert threshold."""
    telemetry = state.get("telemetry_context", {})
    if "error" in telemetry:
        return {
            "route_taken": "routine_clearance (guardrail)",
            "llm_backend": "System Guardrail",
            "final_work_order": f"⚠️ {telemetry['error']}",
        }

    prob_pct = float(telemetry.get("failure_prob_72h", 0.0)) * 100
    msg = f"""### ✅ FleetGuard Routine Clearance — `{telemetry['vehicle_id']}`

- **Asset:** {telemetry.get('make_and_model')} ({telemetry.get('vehicle_type')})
- **Active Corridor:** {telemetry.get('route_corridor')} (Speed: `{telemetry.get('speed_kmh')} km/h`)
- **72h Breakdown Risk:** **{prob_pct:.1f}%** (`{telemetry.get('risk_tier')}` — below `67.0%` workshop threshold)
- **Dispatch Decision:** `CONTINUE_NORMAL_OPERATION` (No €350 workshop bay reservation needed).
"""
    return {
        "route_taken": "routine_clearance",
        "llm_backend": "Conditional Fast-Path (Skipped LLM)",
        "final_work_order": msg,
    }


def node_retrieve_sop_and_roi(state: TriageAgentState) -> dict:
    """Step 2 (Urgent Branch): Query Ollama RAG manuals & compute EUR savings."""
    telemetry = state.get("telemetry_context", {})
    drivers = telemetry.get("top_drivers", [])
    driver_terms = " ".join(f"{d['feature']} {d['value']}" for d in drivers)

    symptom_query = (
        f"{telemetry.get('vehicle_type', '')} {telemetry.get('brake_condition', '')} "
        f"{driver_terms} {telemetry.get('route_corridor', '')}"
    )
    sop_data = json.loads(retrieve_oem_repair_sop.invoke({"symptom_query": symptom_query}))
    roi_data = json.loads(
        calculate_dispatch_roi.invoke(
            {
                "failure_prob_72h": float(telemetry.get("failure_prob_72h", 0.0)),
                "route_corridor": str(telemetry.get("route_corridor", "Unknown")),
            }
        )
    )
    return {"route_taken": "urgent_triage", "sop_context": sop_data, "roi_context": roi_data}


def node_generate_work_order(state: TriageAgentState) -> dict:
    """Step 3 (Urgent Branch): Synthesize grounded Mechanic Work Order with ChatOllama."""
    telemetry = state.get("telemetry_context", {})
    sop_data = state.get("sop_context", {})
    roi_data = state.get("roi_context", {})
    user_query = state.get("user_query", "Generate a 72-hour maintenance triage work order.")

    chat_model, backend_label = _get_ollama_chat_model()
    if chat_model is not None:
        try:
            configure_mlflow()
            import mlflow

            mlflow.langchain.autolog()
            system_prompt = (
                "You are FleetGuard AI, a German commercial fleet maintenance engineering copilot. "
                "Using ONLY the grounded SQLite telemetry, SHAP drivers, retrieved OEM repair SOP chunks, "
                "and ROI metrics provided, write a concise, actionable Workshop Dispatch & Mechanic Work Order "
                "in Markdown with: 1) Vehicle & Risk Summary, 2) Root-Cause SHAP Drivers, "
                "3) Cited OEM Repair Protocol & Parts BOM, and 4) Autobahn Hub Routing & EUR Cost Impact."
            )
            payload = json.dumps(
                {
                    "user_question": user_query,
                    "telemetry": telemetry,
                    "oem_sops": sop_data,
                    "dispatch_roi": roi_data,
                },
                indent=2,
            )
            response = chat_model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=payload)]
            )
            return {"llm_backend": backend_label, "final_work_order": str(response.content)}
        except Exception as exc:
            backend_label = f"Deterministic Fallback (Ollama unreachable: {exc})"

    # Fallback formatter if Ollama daemon isn't running
    drivers_md = "\n".join(
        f"- **{d['feature']}** = `{d['value']}` (SHAP impact: `{d['shap_impact']:+.3f}`)"
        for d in telemetry.get("top_drivers", [])
    ) or "- No elevated SHAP drivers recorded."

    sop_chunks = sop_data.get("technical_sops") or []
    hub_chunks = sop_data.get("depot_routing") or []

    sop_section_md = "\n\n".join(
        f"**[{c.get('doc_id')} — {c.get('section')}]** *(Source: `{c.get('source_file')}`, score `{c.get('relevance_score')}`)*\n{c.get('content')}"
        for c in sop_chunks
    ) or "Run standard multi-point workshop inspection."

    hub_section_md = hub_chunks[0].get("content", "Route to nearest partner depot.") if hub_chunks else ""

    work_order = f"""### 🛠️ FleetGuard Maintenance Work Order — `{telemetry['vehicle_id']}`

**1. Vehicle & 72-Hour Risk Assessment**
- **Asset:** {telemetry.get('make_and_model')} ({telemetry.get('vehicle_type')})
- **Active Corridor:** {telemetry.get('route_corridor')} (Speed: `{telemetry.get('speed_kmh')} km/h`)
- **72h Breakdown Risk:** **{float(telemetry.get('failure_prob_72h', 0))*100:.1f}%** (`{telemetry.get('risk_tier')}`)
- **Dispatch Decision:** `{roi_data.get('recommended_dispatch_action')}`

**2. Top SHAP Root-Cause Drivers**
{drivers_md}

**3. Retrieved OEM Technical Service Bulletins (RAG Citations)**
{sop_section_md}

**4. Autobahn Hub Routing & Financial Impact**
{hub_section_md}

- **Expected Unmitigated Breakdown Loss:** `€{roi_data.get('expected_unmitigated_loss_eur', 0):,.2f}` vs. `€350.00` inspection → **Projected Net Savings: `€{roi_data.get('expected_net_savings_eur', 0):,.2f}`**
"""
    return {"llm_backend": backend_label, "final_work_order": work_order}


# =============================================================================
# 3. GRAPH WIRING & COMPILATION
# =============================================================================
def build_triage_graph():
    builder = StateGraph(TriageAgentState)

    # Register all nodes
    builder.add_node("gather_vehicle_context", node_gather_vehicle_context)
    builder.add_node("retrieve_sop_and_roi", node_retrieve_sop_and_roi)
    builder.add_node("generate_work_order", node_generate_work_order)
    builder.add_node("routine_clearance", node_routine_clearance)

    # Wire edges (including the conditional branch after gather_vehicle_context)
    builder.add_edge(START, "gather_vehicle_context")
    builder.add_conditional_edges(
        "gather_vehicle_context",
        route_by_risk_tier,
        {
            "urgent_triage": "retrieve_sop_and_roi",
            "routine_clearance": "routine_clearance",
        },
    )
    builder.add_edge("retrieve_sop_and_roi", "generate_work_order")
    builder.add_edge("generate_work_order", END)
    builder.add_edge("routine_clearance", END)

    return builder.compile()


TRIAGE_AGENT = build_triage_graph()


def run_fleet_triage_copilot(vehicle_id: str = "", user_query: str = "") -> dict:
    initial_state: TriageAgentState = {
        "user_query": user_query or "Generate 72-hour maintenance triage and workshop dispatch order.",
        "vehicle_id": vehicle_id,
        "telemetry_context": {},
        "sop_context": {},
        "roi_context": {},
        "route_taken": "",
        "llm_backend": "",
        "final_work_order": "",
    }
    return TRIAGE_AGENT.invoke(initial_state)


if __name__ == "__main__":
    print("=== LangGraph Conditional Workflow Diagram ===")
    print(TRIAGE_AGENT.get_graph().draw_ascii())

    print("\n=== Test: High-Risk Truck (TRK-1037) ===")
    res = run_fleet_triage_copilot(vehicle_id="TRK-1037")
    print(f"Branch Taken: {res['route_taken']} | Backend: {res['llm_backend']}\n")
    print(res["final_work_order"])