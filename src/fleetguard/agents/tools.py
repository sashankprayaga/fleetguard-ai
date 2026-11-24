from __future__ import annotations

import json
import pandas as pd
from langchain_core.tools import tool

from fleetguard.rag.retriever import OemManualRetriever
from fleetguard.services.database import get_connection, init_database

# Initialize the Ollama-backed RAG retriever once when the module loads
_RETRIEVER = OemManualRetriever()


@tool
def query_vehicle_telemetry_and_shap(vehicle_id: str) -> str:
    """
    Fetch the latest scored telemetry record, 72-hour breakdown probability,
    risk tier, and top SHAP root-cause drivers for a specific Vehicle_ID (e.g. 'TRK-1037').
    """
    init_database()
    clean_id = vehicle_id.strip().upper()
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM fleet_predictions WHERE UPPER(vehicle_id) = ? ORDER BY id DESC LIMIT 1",
            conn,
            params=(clean_id,),
        )

    if df.empty:
        return json.dumps({"error": f"No telemetry found for vehicle_id '{clean_id}'."})

    row = df.iloc[0].to_dict()
    row["top_drivers"] = json.loads(row.pop("top_drivers_json", "[]") or "[]")
    return json.dumps(row, default=str)


@tool
def list_top_flagged_vehicles(limit: int = 5) -> str:
    """
    List the highest-risk trucks currently flagged for 72-hour preventive maintenance
    in the SQLite fleet telemetry database.
    """
    init_database()
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT vehicle_id, vehicle_type, make_and_model, route_corridor,
                   failure_prob_72h, risk_tier, days_since_last_maintenance,
                   brake_condition, oil_quality, engine_temperature, vibration_levels
            FROM fleet_predictions
            WHERE id IN (SELECT MAX(id) FROM fleet_predictions GROUP BY vehicle_id)
            ORDER BY failure_prob_72h DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )
    return df.to_json(orient="records")


@tool
def retrieve_oem_repair_sop(symptom_query: str) -> str:
    """
    Retrieve OEM diagnostic steps, repair actions, workshop duration, and parts kit IDs
    matching a truck's sensor anomalies or SHAP drivers using Ollama semantic embeddings.
    """
    matches = _RETRIEVER.retrieve(symptom_query, top_k=2)
    hub_match = _RETRIEVER.retrieve("German Autobahn Regional Service Depot", top_k=1)
    return json.dumps({"technical_sops": matches, "depot_routing": hub_match})


@tool
def calculate_dispatch_roi(failure_prob_72h: float, route_corridor: str) -> str:
    """
    Calculate expected financial savings (EUR) and dispatch action priority comparing
    a €350 preventive workshop inspection against a €4,500 Autobahn roadside breakdown.
    """
    prob = max(0.0, min(1.0, float(failure_prob_72h)))
    expected_breakdown_loss = prob * 4500.0
    inspection_cost = 350.0
    expected_net_benefit = expected_breakdown_loss - inspection_cost

    if prob >= 0.80:
        action = "IMMEDIATE_PULL_OVER_NEXT_HUB"
    elif prob >= 0.67:
        action = "SCHEDULE_DESTINATION_DEPOT_WITHIN_24H"
    elif prob >= 0.45:
        action = "MONITOR_AND_PREBOOK_48H_SLOT"
    else:
        action = "CONTINUE_NORMAL_OPERATION"

    return json.dumps(
        {
            "route_corridor": route_corridor,
            "failure_prob_72h": round(prob, 4),
            "expected_unmitigated_loss_eur": round(expected_breakdown_loss, 2),
            "preventive_inspection_cost_eur": inspection_cost,
            "expected_net_savings_eur": round(expected_net_benefit, 2),
            "recommended_dispatch_action": action,
        }
    )