from __future__ import annotations

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fleetguard.config.settings import PATHS
from fleetguard.services.database import get_connection, init_database
from fleetguard.services.inference import score_fleet_telemetry

st.set_page_config(
    page_title="FleetGuard AI | German Fleet Predictive Maintenance",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "CRITICAL": "#DC2626",
    "HIGH": "#F97316",
    "ELEVATED": "#EAB308",
    "NORMAL": "#16A34A",
}


@st.cache_data(ttl=300)
def load_historical_telemetry() -> pd.DataFrame:
    return pd.read_parquet(PATHS.raw_data / "synthetic_fleet_history.parquet")


@st.cache_data(ttl=300)
def load_model_artifacts() -> tuple[dict, pd.DataFrame]:
    with open(PATHS.models_dir / "champion_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(PATHS.models_dir / "global_shap_importance.json", encoding="utf-8") as f:
        shap_data = json.load(f)
    shap_df = pd.DataFrame(shap_data["global_feature_importance"])
    return meta, shap_df


def get_initial_live_snapshot(raw_df: pd.DataFrame, step_offset: int = 0) -> tuple[pd.DataFrame, dict]:
    """Score a 100-truck fleet snapshot at a given historical offset for interactive simulation."""
    unique_ts = sorted(raw_df["Timestamp"].unique())
    target_idx = max(10, len(unique_ts) - 1 - step_offset)
    target_ts = unique_ts[target_idx]
    window_start = unique_ts[max(0, target_idx - 5)]

    fleet_ids = sorted(raw_df["Vehicle_ID"].unique())[:100]
    window_df = raw_df[
        (raw_df["Vehicle_ID"].isin(fleet_ids))
        & (raw_df["Timestamp"] >= window_start)
        & (raw_df["Timestamp"] <= target_ts)
    ].copy()

    scored_window, summary = score_fleet_telemetry(
        window_df,
        source_override="synthetic_live_stream",
        attach_shap_for_alerts=True,
        persist_to_db=True,
        log_to_mlflow=True,
    )
    latest_tick = scored_window[scored_window["Timestamp"] == target_ts].copy()
    return latest_tick, summary


def main() -> None:
    init_database()
    raw_df = load_historical_telemetry()
    meta, shap_df = load_model_artifacts()

    if "sim_offset" not in st.session_state:
        st.session_state.sim_offset = 0
    if "live_snapshot" not in st.session_state:
        snap, summ = get_initial_live_snapshot(raw_df, step_offset=0)
        st.session_state.live_snapshot = snap
        st.session_state.live_summary = summ

    # Sidebar Controls
    with st.sidebar:
        st.title("🚛 FleetGuard AI")
        st.caption("72-Hour Predictive Maintenance & MLOps Console")
        st.divider()

        st.markdown(f"**Champion Model:** `{meta['champion_model'].upper()}`")
        st.markdown(f"**Decision Threshold:** `{meta['optimal_threshold']}` (12% Bay Cap)")
        st.markdown(f"**Test PR-AUC:** `{meta['champion_heldout_test_metrics']['pr_auc']}`")
        st.markdown(f"**Test Precision:** `{meta['champion_heldout_test_metrics']['precision']*100:.1f}%`")

        st.divider()
        if st.button("🔄 Simulate Next 4h Telemetry Tick", width="stretch", type="primary"):
            st.session_state.sim_offset = (st.session_state.sim_offset + 1) % 30
            with st.spinner("Scoring 100-truck telemetry packet & running SHAP + PSI drift audit..."):
                snap, summ = get_initial_live_snapshot(raw_df, step_offset=st.session_state.sim_offset)
                st.session_state.live_snapshot = snap
                st.session_state.live_summary = summ

        corridor_options = ["All Corridors"] + sorted(st.session_state.live_snapshot["Route_Corridor"].unique().tolist())
        selected_corridor = st.selectbox("Filter Autobahn Corridor", corridor_options)

        risk_options = ["CRITICAL", "HIGH", "ELEVATED", "NORMAL"]
        selected_tiers = st.multiselect("Risk Tiers", risk_options, default=risk_options)

    snap_df: pd.DataFrame = st.session_state.live_snapshot
    summary: dict = st.session_state.live_summary

    filtered_df = snap_df[snap_df["Risk_Tier"].isin(selected_tiers)].copy()
    if selected_corridor != "All Corridors":
        filtered_df = filtered_df[filtered_df["Route_Corridor"] == selected_corridor]

    tab_live, tab_batch, tab_model, tab_mlops = st.tabs(
        [
            "🗺️ Live Fleet Command",
            "📤 Batch Telemetry Scoring",
            "🧠 Model & SHAP Explainability",
            "📡 MLOps Drift & Audit Logs",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: LIVE FLEET COMMAND
    # -------------------------------------------------------------------------
    with tab_live:
        st.subheader(f"Active German Autobahn Telemetry — Timestamp: `{snap_df['Timestamp'].iloc[0]}`")

        alerts_df = snap_df[snap_df["Alert_Triggered"] == 1]
        critical_count = int((snap_df["Risk_Tier"] == "CRITICAL").sum())
        alert_rate = float(snap_df["Alert_Triggered"].mean() * 100)
        est_net_savings = len(alerts_df) * 0.70 * (4500.0 - 350.0) - len(alerts_df) * 0.30 * 350.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monitored Active Fleet", f"{len(snap_df)} Trucks")
        c2.metric("72h Workshop Alerts", f"{len(alerts_df)} Trucks", delta=f"{critical_count} CRITICAL", delta_color="inverse")
        c3.metric("Workshop Bay Load", f"{alert_rate:.1f}%", delta="12.0% Max Cap", delta_color="off")
        c4.metric("Est. Batch Net Savings", f"€{est_net_savings:,.0f}", delta="vs. Reactive Towing")

        col_map, col_queue = st.columns([1.35, 1.0])

        with col_map:
            st.markdown("#### German Autobahn Live Risk Map")
            fig_map = px.scatter_map(
                filtered_df,
                lat="Latitude",
                lon="Longitude",
                color="Risk_Tier",
                color_discrete_map=RISK_COLORS,
                size="Failure_Prob_72h",
                size_max=16,
                hover_name="Vehicle_ID",
                hover_data={
                    "Vehicle_Type": True,
                    "Route_Corridor": True,
                    "Failure_Prob_72h": ":.1%",
                    "Engine_Temperature": ":.1f",
                    "Oil_Quality": ":.1f",
                    "Latitude": False,
                    "Longitude": False,
                },
                zoom=5.1,
                center={"lat": 51.1657, "lon": 10.4515},
                map_style="carto-positron",
                height=520,
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Risk Tier")
            st.plotly_chart(fig_map, width="stretch")

        with col_queue:
            st.markdown("#### Priority Workshop Dispatch Queue")
            queue_df = filtered_df.sort_values("Failure_Prob_72h", ascending=False)
            display_cols = [
                "Vehicle_ID",
                "Vehicle_Type",
                "Route_Corridor",
                "Failure_Prob_72h",
                "Risk_Tier",
            ]
            st.dataframe(
                queue_df[display_cols].style.format({"Failure_Prob_72h": "{:.1%}"}),
                width="stretch",
                height=260,
                hide_index=True,
            )

            st.markdown("#### 🔍 Per-Truck SHAP Root-Cause Inspector")
            flagged_trucks = queue_df[queue_df["Alert_Triggered"] == 1]["Vehicle_ID"].tolist()
            if flagged_trucks:
                chosen_id = st.selectbox("Select Flagged Truck for Diagnostic Drivers", flagged_trucks)
                truck_row = queue_df[queue_df["Vehicle_ID"] == chosen_id].iloc[0]
                drivers = json.loads(truck_row["Top_Drivers_JSON"])
                st.info(
                    f"**{chosen_id}** ({truck_row['Make_and_Model']} — {truck_row['Route_Corridor']}) | "
                    f"72h Risk: **{truck_row['Failure_Prob_72h']:.1%}** ({truck_row['Risk_Tier']})"
                )
                for drv in drivers:
                    st.write(f"- **{drv['feature']}** = `{drv['value']}` → SHAP impact `{drv['shap_impact']:+.3f}`")
            else:
                st.success("No trucks exceed the 72h alert threshold in the current filter.")

    # -------------------------------------------------------------------------
    # TAB 2: BATCH CSV / PARQUET SCORING
    # -------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Upload Fleet Telemetry Batch for 72-Hour Risk Scoring")
        st.write(
            "Upload a `.csv` or `.parquet` file conforming to the FleetGuard telemetry schema. "
            "Incoming rows are validated via Pydantic before XGBoost + SHAP inference."
        )

        sample_export = raw_df.tail(150).drop(columns=["Maintenance_Required"], errors="ignore")
        st.download_button(
            label="⬇️ Download Sample Unlabeled Telemetry CSV (150 rows)",
            data=sample_export.to_csv(index=False).encode("utf-8"),
            file_name="sample_german_fleet_telemetry.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Choose telemetry file", type=["csv", "parquet"])
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".parquet"):
                    up_df = pd.read_parquet(uploaded)
                else:
                    up_df = pd.read_csv(uploaded)
                    up_df["Timestamp"] = pd.to_datetime(up_df["Timestamp"])

                with st.spinner("Validating schema & scoring uploaded batch..."):
                    scored_upload, up_summary = score_fleet_telemetry(
                        up_df,
                        source_override="uploaded_batch",
                        attach_shap_for_alerts=True,
                        persist_to_db=True,
                        log_to_mlflow=True,
                    )

                st.success(
                    f"Scored {up_summary['batch_size']} rows | Alerts triggered: {up_summary['alert_count']} "
                    f"({up_summary['alert_rate_pct']}%) | Drift Status: {up_summary['drift_status']}"
                )
                st.dataframe(
                    scored_upload[
                        [
                            "Vehicle_ID",
                            "Timestamp",
                            "Vehicle_Type",
                            "Route_Corridor",
                            "Failure_Prob_72h",
                            "Risk_Tier",
                            "Top_Drivers_JSON",
                        ]
                    ].sort_values("Failure_Prob_72h", ascending=False),
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    label="⬇️ Download Scored Predictions CSV",
                    data=scored_upload.to_csv(index=False).encode("utf-8"),
                    file_name="fleetguard_scored_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch validation/scoring failed: {exc}")

    # -------------------------------------------------------------------------
    # TAB 3: MODEL & SHAP EXPLAINABILITY
    # -------------------------------------------------------------------------
    with tab_model:
        st.subheader("Champion Model Benchmark & Global SHAP Feature Attribution")
        test_m = meta["champion_heldout_test_metrics"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Held-Out PR-AUC", f"{test_m['pr_auc']:.4f}", delta="vs. 0.3176 Base Rate")
        m2.metric("Held-Out Precision", f"{test_m['precision']*100:.2f}%")
        m3.metric("Held-Out Recall", f"{test_m['recall']*100:.2f}%")
        m4.metric("Held-Out Net Savings", f"€{test_m['net_savings_vs_reactive_eur']:,.0f}")

        st.markdown("#### Top 15 Global Risk Drivers (Mean |SHAP| on Held-Out Test Set)")
        top15 = shap_df.head(15).iloc[::-1]
        fig_shap = px.bar(
            top15,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            text="relative_importance_pct",
            labels={"mean_abs_shap": "Mean |SHAP| Impact (Log-Odds)", "feature": "Engineered / Sensor Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig_shap.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_shap.update_layout(showlegend=False, coloraxis_showscale=False, height=520)
        st.plotly_chart(fig_shap, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 4: MLOPS DRIFT & AUDIT LOGS
    # -------------------------------------------------------------------------
    with tab_mlops:
        st.subheader("Production Covariate Drift (PSI) & MLflow Monitoring Logs")
        st.code("mlflow ui --backend-store-uri sqlite:///data/database/mlflow.db", language="bash")

        d1, d2, d3 = st.columns(3)
        d1.metric("Latest Batch Drift Status", summary["drift_status"])
        d2.metric("Max Feature PSI", f"{summary['max_psi']:.4f}", delta="Alert > 0.25", delta_color="inverse")
        d3.metric("Latest MLflow Run ID", summary.get("mlflow_run_id", "N/A")[:12] + "...")

        if summary.get("feature_psi"):
            psi_df = pd.DataFrame(
                [{"Feature": k, "PSI": v} for k, v in summary["feature_psi"].items()]
            ).sort_values("PSI", ascending=True)
            fig_psi = go.Figure(
                go.Bar(
                    x=psi_df["PSI"],
                    y=psi_df["Feature"],
                    orientation="h",
                    marker_color=[
                        "#DC2626" if v >= 0.25 else "#F97316" if v >= 0.10 else "#16A34A"
                        for v in psi_df["PSI"]
                    ],
                )
            )
            fig_psi.add_vline(x=0.10, line_dash="dash", line_color="#F97316", annotation_text="Moderate (0.10)")
            fig_psi.add_vline(x=0.25, line_dash="dash", line_color="#DC2626", annotation_text="Critical (0.25)")
            fig_psi.update_layout(title="Population Stability Index (PSI) vs. Training Reference", height=360)
            st.plotly_chart(fig_psi, width="stretch")

        st.markdown("#### Recent SQLite Monitoring Audit Trail (`monitoring_logs`)")
        with get_connection() as conn:
            logs_df = pd.read_sql_query(
                "SELECT id, logged_at, data_source, batch_size, alert_count, alert_rate_pct, "
                "avg_failure_prob, drift_status, max_psi, mlflow_run_id "
                "FROM monitoring_logs ORDER BY id DESC LIMIT 15",
                conn,
            )
        st.dataframe(logs_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()