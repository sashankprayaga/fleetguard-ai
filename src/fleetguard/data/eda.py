from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from fleetguard.config.settings import PATHS
from fleetguard.data.validator import validate_telemetry_dataframe

# Explicit feature governance: prevents ID, timestamp, or GPS memorization in ML models
IDENTIFIER_COLS = ["Vehicle_ID", "Timestamp"]
GEOSPATIAL_COLS = ["Route_Corridor", "Route_Progress", "Latitude", "Longitude", "Speed_kmh"]
METADATA_COLS = ["Data_Source"]
TARGET_COL = "Maintenance_Required"

NUMERICAL_FEATURES = [
    "Year_of_Manufacture",
    "Usage_Hours",
    "Load_Capacity",
    "Actual_Load",
    "Engine_Temperature",
    "Tire_Pressure",
    "Fuel_Consumption",
    "Battery_Status",
    "Vibration_Levels",
    "Oil_Quality",
    "Failure_History",
    "Days_Since_Last_Maintenance",
    "PCR",
    "UIR",
    "TPI",
]

CATEGORICAL_FEATURES = [
    "Make_and_Model",
    "Vehicle_Type",
    "Last_Maintenance_Type",
    "Brake_Condition",
]


def run_exploratory_audit(df: pd.DataFrame) -> dict:
    """Generate a structured EDA and feature governance summary."""
    validation = validate_telemetry_dataframe(df, require_target=True)

    # Verify every column is accounted for (zero unreviewed columns)
    accounted_cols = set(
        IDENTIFIER_COLS
        + GEOSPATIAL_COLS
        + METADATA_COLS
        + NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COL]
    )
    unaccounted_cols = sorted(set(df.columns) - accounted_cols)

    class_counts = df[TARGET_COL].value_counts().to_dict()
    positive_rate = float(df[TARGET_COL].mean())

    by_vehicle_type = (
        df.groupby("Vehicle_Type")
        .agg(
            trucks=("Vehicle_ID", "nunique"),
            records=("Vehicle_ID", "count"),
            failure_rate=(TARGET_COL, "mean"),
            avg_engine_temp=("Engine_Temperature", "mean"),
            avg_oil_quality=("Oil_Quality", "mean"),
            avg_vibration=("Vibration_Levels", "mean"),
        )
        .round(4)
        .to_dict(orient="index")
    )

    numeric_summary = df[NUMERICAL_FEATURES].describe().loc[["mean", "std", "min", "50%", "max"]].round(3).to_dict()

    return {
        "validation_passed": validation.is_valid,
        "validation_errors": validation.sample_row_errors,
        "total_rows": int(len(df)),
        "unique_vehicles": int(df["Vehicle_ID"].nunique()),
        "time_range": {
            "start": str(df["Timestamp"].min()),
            "end": str(df["Timestamp"].max()),
        },
        "target_distribution": {
            "healthy_0": int(class_counts.get(0, 0)),
            "maintenance_required_1": int(class_counts.get(1, 0)),
            "positive_rate_pct": round(positive_rate * 100, 2),
        },
        "feature_governance": {
            "ml_numerical_features": NUMERICAL_FEATURES,
            "ml_categorical_features": CATEGORICAL_FEATURES,
            "excluded_from_ml": IDENTIFIER_COLS + GEOSPATIAL_COLS + METADATA_COLS,
            "unaccounted_columns": unaccounted_cols,
        },
        "summary_by_vehicle_type": by_vehicle_type,
        "numeric_feature_stats": numeric_summary,
    }


def save_eda_report(report: dict, output_path: Path | None = None) -> Path:
    PATHS.processed_data.mkdir(parents=True, exist_ok=True)
    target = output_path or (PATHS.processed_data / "eda_audit_report.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return target


if __name__ == "__main__":
    raw_path = PATHS.raw_data / "synthetic_fleet_history.parquet"
    df = pd.read_parquet(raw_path)
    audit = run_exploratory_audit(df)
    saved_to = save_eda_report(audit)
    print(f"EDA audit saved to: {saved_to}")
    print(json.dumps({k: audit[k] for k in ["validation_passed", "total_rows", "unique_vehicles", "target_distribution"]}, indent=2))
    print("Summary by Vehicle Type:")
    print(pd.DataFrame(audit["summary_by_vehicle_type"]).T)