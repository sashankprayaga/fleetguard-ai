from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from fleetguard.config.settings import PATHS
from fleetguard.data.validator import validate_telemetry_dataframe
from fleetguard.explainability.shap_explainer import FleetShapExplainer
from fleetguard.features.engineering import ALL_MODEL_FEATURES, add_domain_and_rolling_features
from fleetguard.mlops.tracker import evaluate_feature_drift, log_monitoring_snapshot_to_mlflow
from fleetguard.services.database import save_scored_batch


def classify_risk_tier(prob: float, threshold: float) -> str:
    if prob >= max(threshold + 0.13, 0.80):
        return "CRITICAL"
    if prob >= threshold:
        return "HIGH"
    if prob >= threshold * 0.65:
        return "ELEVATED"
    return "NORMAL"


def score_fleet_telemetry(
    df: pd.DataFrame,
    source_override: str | None = None,
    attach_shap_for_alerts: bool = True,
    persist_to_db: bool = True,
    log_to_mlflow: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate incoming fleet telemetry, engineer features, predict 72h failure risk,
    attach SHAP root-cause explanations for flagged trucks, run PSI drift audit,
    and persist results to SQLite + MLflow.
    """
    working_df = df.copy()
    if source_override:
        working_df["Data_Source"] = source_override

    validation = validate_telemetry_dataframe(working_df, require_target=False)
    if not validation.is_valid:
        raise ValueError(f"Incoming telemetry failed validation: {validation.sample_row_errors}")

    with open(PATHS.models_dir / "champion_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["optimal_threshold"])
    champion_pipe = joblib.load(PATHS.models_dir / "champion_model.joblib")

    enriched = add_domain_and_rolling_features(working_df)
    probs = champion_pipe.predict_proba(enriched[ALL_MODEL_FEATURES])[:, 1]

    enriched["Failure_Prob_72h"] = np.round(probs, 4)
    enriched["Alert_Triggered"] = (probs >= threshold).astype(int)
    enriched["Risk_Tier"] = [classify_risk_tier(float(p), threshold) for p in probs]
    enriched["Top_Drivers_JSON"] = "[]"

    # Attach local SHAP root-cause explanations for all high-risk/critical trucks
    alert_mask = enriched["Alert_Triggered"] == 1
    if attach_shap_for_alerts and alert_mask.any():
        explainer = FleetShapExplainer(champion_pipe)
        alert_explanations = explainer.explain_vehicles(enriched.loc[alert_mask], top_k=3)
        driver_json_list = [json.dumps(exp["top_risk_drivers"]) for exp in alert_explanations]
        enriched.loc[alert_mask, "Top_Drivers_JSON"] = driver_json_list

    # Drift audit against the training portion of historical data
    raw_hist_path = PATHS.raw_data / "synthetic_fleet_history.parquet"
    if raw_hist_path.exists():
        ref_df = pd.read_parquet(raw_hist_path)
        ref_train = ref_df.iloc[: int(len(ref_df) * 0.70)]
        drift_report = evaluate_feature_drift(ref_train, enriched)
    else:
        drift_report = {"drift_status": "HEALTHY", "max_psi": 0.0, "feature_psi": {}}

    source_tag = str(enriched["Data_Source"].iloc[0])
    batch_summary = {
        "data_source": source_tag,
        "batch_size": int(len(enriched)),
        "alert_count": int(alert_mask.sum()),
        "alert_rate_pct": round(float(alert_mask.mean() * 100), 2),
        "avg_failure_prob": round(float(probs.mean()), 4),
        "optimal_threshold": threshold,
        "drift_status": drift_report["drift_status"],
        "max_psi": drift_report["max_psi"],
        "feature_psi": drift_report["feature_psi"],
    }

    if log_to_mlflow:
        run_id = log_monitoring_snapshot_to_mlflow(drift_report, batch_summary, source_tag=source_tag)
        batch_summary["mlflow_run_id"] = run_id

    if persist_to_db:
        save_scored_batch(enriched, batch_summary)

    return enriched, batch_summary


if __name__ == "__main__":
    raw_df = pd.read_parquet(PATHS.raw_data / "synthetic_fleet_history.parquet")
    latest_ts = raw_df["Timestamp"].max()
    sample_vehicles = sorted(raw_df["Vehicle_ID"].unique())[:50]
    recent_window = raw_df[
        (raw_df["Vehicle_ID"].isin(sample_vehicles))
        & (raw_df["Timestamp"] >= latest_ts - pd.Timedelta(hours=20))
    ].copy()

    scored_df, summary = score_fleet_telemetry(
        recent_window,
        source_override="synthetic_live_stream",
        attach_shap_for_alerts=True,
        persist_to_db=True,
        log_to_mlflow=True,
    )
    latest_scored = scored_df[scored_df["Timestamp"] == latest_ts]
    print("Batch Monitoring Summary:")
    print(json.dumps(summary, indent=2))
    print("\nTop 5 Highest-Risk Trucks Right Now:")
    print(
        latest_scored.sort_values("Failure_Prob_72h", ascending=False)[
            ["Vehicle_ID", "Vehicle_Type", "Route_Corridor", "Failure_Prob_72h", "Risk_Tier"]
        ].head(5)
    )