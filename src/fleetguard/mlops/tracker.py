from __future__ import annotations

import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from fleetguard.config.settings import PATHS
from fleetguard.features.engineering import add_domain_and_rolling_features

DRIFT_MONITORED_FEATURES = [
    "Days_Since_Last_Maintenance",
    "UIR",
    "Oil_Quality",
    "Vibration_Roll6_Mean",
    "Engine_Temp_Roll6_Mean",
]


def configure_mlflow() -> str:
    """Point MLflow to the local SQLite tracking store and artifact directory."""
    PATHS.database_dir.mkdir(parents=True, exist_ok=True)
    PATHS.mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{PATHS.mlflow_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between reference training distribution
    and incoming telemetry distribution.
    < 0.10: No significant drift | 0.10-0.25: Moderate drift | > 0.25: Critical drift
    """
    ref_clean = reference.dropna().to_numpy(dtype=float)
    cur_clean = current.dropna().to_numpy(dtype=float)
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(ref_clean, quantiles))
    if len(bin_edges) < 3:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    eps = 1e-6
    ref_pct = np.clip(ref_counts / max(len(ref_clean), 1), eps, 1.0)
    cur_pct = np.clip(cur_counts / max(len(cur_clean), 1), eps, 1.0)

    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi_val, 4)


def evaluate_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Compute PSI across top SHAP features and classify overall fleet drift status."""
    ref_enriched = (
        reference_df
        if all(c in reference_df.columns for c in DRIFT_MONITORED_FEATURES)
        else add_domain_and_rolling_features(reference_df)
    )
    cur_enriched = (
        current_df
        if all(c in current_df.columns for c in DRIFT_MONITORED_FEATURES)
        else add_domain_and_rolling_features(current_df)
    )

    feature_psi: dict[str, float] = {}
    for feat in DRIFT_MONITORED_FEATURES:
        feature_psi[feat] = compute_psi(ref_enriched[feat], cur_enriched[feat])

    max_psi = max(feature_psi.values()) if feature_psi else 0.0
    if max_psi >= 0.25:
        status = "CRITICAL_DRIFT"
    elif max_psi >= 0.10:
        status = "MODERATE_DRIFT"
    else:
        status = "HEALTHY"

    return {
        "drift_status": status,
        "max_psi": round(max_psi, 4),
        "feature_psi": feature_psi,
    }


def log_champion_to_mlflow() -> dict:
    """Log saved Phase 4 & Phase 5 champion artifacts and metrics into MLflow + Model Registry."""
    tracking_uri = configure_mlflow()
    meta_path = PATHS.models_dir / "champion_metadata.json"
    model_path = PATHS.models_dir / "champion_model.joblib"
    shap_path = PATHS.models_dir / "global_shap_importance.json"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    champion_pipe = joblib.load(model_path)
    mlflow.set_experiment("FleetGuard-72h-Predictive-Maintenance")

    with mlflow.start_run(run_name=f"champion-{meta['champion_model']}-72h") as run:
        mlflow.log_params(
            {
                "champion_model": meta["champion_model"],
                "prediction_horizon_hours": meta["prediction_horizon_hours"],
                "optimal_threshold": meta["optimal_threshold"],
                "feature_count": meta["feature_count"],
                "max_workshop_capacity_pct": 12.0,
            }
        )

        test_m = meta["champion_heldout_test_metrics"]
        mlflow.log_metrics(
            {
                "test_roc_auc": test_m["roc_auc"],
                "test_pr_auc": test_m["pr_auc"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_f1": test_m["f1"],
                "test_alert_rate_pct": test_m["alert_rate_pct"],
                "test_net_savings_eur": test_m["net_savings_vs_reactive_eur"],
            }
        )

        mlflow.log_artifact(str(meta_path), artifact_path="metadata")
        if shap_path.exists():
            mlflow.log_artifact(str(shap_path), artifact_path="explainability")

        mlflow.sklearn.log_model(
            sk_model=champion_pipe,
            name="model",
            registered_model_name="fleetguard-72h-predictor",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        )

        return {
            "tracking_uri": tracking_uri,
            "run_id": run.info.run_id,
            "registered_model": "fleetguard-72h-predictor",
            "champion_model": meta["champion_model"],
        }


def log_monitoring_snapshot_to_mlflow(
    drift_report: dict,
    batch_metrics: dict,
    source_tag: str = "synthetic_live_stream",
) -> str:
    """Log live inference alert rates and PSI drift metrics to MLflow monitoring experiment."""
    configure_mlflow()
    mlflow.set_experiment("FleetGuard-Production-Monitoring")

    with mlflow.start_run(run_name=f"monitor-{source_tag}") as run:
        mlflow.log_param("data_source", source_tag)
        mlflow.log_param("drift_status", drift_report["drift_status"])
        mlflow.log_metric("max_psi", drift_report["max_psi"])
        for feat, psi_val in drift_report["feature_psi"].items():
            mlflow.log_metric(f"psi_{feat}", psi_val)

        for k, v in batch_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        return run.info.run_id


if __name__ == "__main__":
    result = log_champion_to_mlflow()
    print(json.dumps(result, indent=2))