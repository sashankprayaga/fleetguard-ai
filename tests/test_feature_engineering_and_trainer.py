from fleetguard.data.synthetic_generator import generate_historical_telemetry
from fleetguard.features.engineering import (
    ALL_MODEL_FEATURES,
    HORIZON_TARGET_COL,
    add_domain_and_rolling_features,
    add_forward_failure_horizon,
)
from fleetguard.models.trainer import find_cost_optimal_threshold, train_champion_models
import pandas as pd
import numpy as np


def test_rolling_features_are_strictly_causal():
    df = generate_historical_telemetry(num_vehicles=2, steps_per_vehicle=10, seed=11)
    first_pass = add_domain_and_rolling_features(df)

    df_modified = df.copy()
    v1_mask = df_modified["Vehicle_ID"] == "TRK-1001"
    last_idx = df_modified[v1_mask].index[-1]
    df_modified.loc[last_idx, "Engine_Temperature"] = 140.0

    second_pass = add_domain_and_rolling_features(df_modified)
    v1_early_1 = first_pass[first_pass["Vehicle_ID"] == "TRK-1001"].iloc[:-1]["Engine_Temp_Roll6_Mean"]
    v1_early_2 = second_pass[second_pass["Vehicle_ID"] == "TRK-1001"].iloc[:-1]["Engine_Temp_Roll6_Mean"]
    assert np.allclose(v1_early_1.values, v1_early_2.values)
    assert all(col in first_pass.columns for col in ALL_MODEL_FEATURES)


def test_forward_horizon_labeling_excludes_censored_tail():
    df = generate_historical_telemetry(num_vehicles=2, steps_per_vehicle=15, seed=11)
    labeled = add_forward_failure_horizon(df, horizon_steps=5)
    v1 = labeled[labeled["Vehicle_ID"] == "TRK-1001"].sort_values("Timestamp")
    assert v1[HORIZON_TARGET_COL].iloc[-5:].isna().all()
    assert v1[HORIZON_TARGET_COL].iloc[:-5].notna().all()


def test_cost_optimizer_prefers_lower_threshold_for_expensive_breakdowns():
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.02, 0.05, 0.12, 0.25, 0.28, 0.75])
    opt = find_cost_optimal_threshold(y_true, y_prob, inspection_cost=350.0, breakdown_cost=4500.0, max_alert_rate=0.5)
    assert opt["optimal_threshold"] <= 0.28
    assert opt["validation_net_savings_eur"] > 0


def test_champion_trainer_smoke():
    df = generate_historical_telemetry(num_vehicles=30, steps_per_vehicle=60, seed=42)
    report = train_champion_models(df, horizon_steps=6, save_artifacts=False)
    assert report["champion_model"] in {"xgboost", "lightgbm"}
    assert "champion_heldout_test_metrics" in report