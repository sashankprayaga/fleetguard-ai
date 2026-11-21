from __future__ import annotations

import numpy as np
import pandas as pd

from fleetguard.data.eda import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# Nominal operating temperature (°C) and battery baseline (%) by Vehicle_Type
POWERTRAIN_BASELINES: dict[str, tuple[float, float]] = {
    "Heavy Haul Truck": (83.0, 60.0),
    "Refrigerated Truck": (83.0, 60.0),
    "Tanker Truck": (84.2, 60.0),
    "Flatbed Carrier": (81.0, 60.0),
    "Regional Box Truck": (80.0, 60.0),
    "Electric Semi": (65.2, 75.0),
}

ENGINEERED_NUMERICAL_FEATURES = [
    "Thermal_Excess_C",
    "Battery_Deficit_Pct",
    "Oil_Degradation_Rate",
    "Engine_Temp_Roll6_Mean",
    "Engine_Temp_Delta_24h",
    "Vibration_Roll6_Mean",
    "Vibration_Delta_24h",
    "Mechanical_Stress_Index",
]

ALL_MODEL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
ALL_MODEL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
ALL_MODEL_FEATURES = ALL_MODEL_NUMERICAL_FEATURES + ALL_MODEL_CATEGORICAL_FEATURES


def add_domain_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add powertrain-normalized sensor deltas and strictly causal (past-only)
    24-hour rolling telemetry features per Vehicle_ID.
    """
    out = df.sort_values(["Vehicle_ID", "Timestamp"]).reset_index(drop=True).copy()

    # 1. Powertrain-relative baselines (row-level, zero leakage)
    nominal_temp = out["Vehicle_Type"].map(lambda vt: POWERTRAIN_BASELINES.get(vt, (82.0, 60.0))[0])
    nominal_batt = out["Vehicle_Type"].map(lambda vt: POWERTRAIN_BASELINES.get(vt, (82.0, 60.0))[1])

    out["Thermal_Excess_C"] = np.round(out["Engine_Temperature"] - nominal_temp, 3)
    out["Battery_Deficit_Pct"] = np.round(np.maximum(0.0, nominal_batt - out["Battery_Status"]), 3)
    out["Oil_Degradation_Rate"] = np.round(
        (100.0 - out["Oil_Quality"]) / (out["Days_Since_Last_Maintenance"] + 1.0), 4
    )

    # 2. Causal per-vehicle rolling window (6 steps * 4h = 24 hours trailing)
    grouped = out.groupby("Vehicle_ID", sort=False)
    out["Engine_Temp_Roll6_Mean"] = np.round(
        grouped["Engine_Temperature"].transform(lambda s: s.rolling(6, min_periods=1).mean()), 3
    )
    out["Engine_Temp_Delta_24h"] = np.round(
        out["Engine_Temperature"] - grouped["Engine_Temperature"].transform(lambda s: s.shift(5).fillna(s.iloc[0])),
        3,
    )
    out["Vibration_Roll6_Mean"] = np.round(
        grouped["Vibration_Levels"].transform(lambda s: s.rolling(6, min_periods=1).mean()), 3
    )
    out["Vibration_Delta_24h"] = np.round(
        out["Vibration_Levels"] - grouped["Vibration_Levels"].transform(lambda s: s.shift(5).fillna(s.iloc[0])),
        3,
    )

    # 3. Composite mechanical stress index
    out["Mechanical_Stress_Index"] = np.round(
        (np.maximum(0.0, out["Thermal_Excess_C"]) * 0.12)
        + (out["Vibration_Roll6_Mean"] * 0.85)
        + ((100.0 - out["Oil_Quality"]) * 0.05)
        + (out["PCR"] * 1.2),
        3,
    )

    # Restore global chronological order
    out = out.sort_values(["Timestamp", "Vehicle_ID"]).reset_index(drop=True)
    return out

HORIZON_TARGET_COL = "Failure_Within_72h"


def add_forward_failure_horizon(
    df: pd.DataFrame,
    horizon_steps: int = 18,
    target_col: str = "Maintenance_Required",
) -> pd.DataFrame:
    """
    Create a forward-looking early-warning binary target (1 if the vehicle fails
    within the next `horizon_steps` observations, e.g., 18 * 4h = 72 hours).
    Rows at the tail of each vehicle's history where the full horizon is unobserved
    are marked NaN so training/evaluation never treat censored tails as healthy.
    """
    out = df.sort_values(["Vehicle_ID", "Timestamp"]).reset_index(drop=True).copy()

    def _future_any_failure(series: pd.Series) -> pd.Series:
        # Look strictly forward steps 1..horizon_steps
        future_max = (
            series.shift(-1)
            .iloc[::-1]
            .rolling(window=horizon_steps, min_periods=horizon_steps)
            .max()
            .iloc[::-1]
        )
        return future_max

    out[HORIZON_TARGET_COL] = out.groupby("Vehicle_ID", sort=False)[target_col].transform(_future_any_failure)
    out = out.sort_values(["Timestamp", "Vehicle_ID"]).reset_index(drop=True)
    return out