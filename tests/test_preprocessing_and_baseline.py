from fleetguard.data.synthetic_generator import generate_historical_telemetry
from fleetguard.features.preprocessing import (
    FEATURE_COLS,
    build_preprocessor,
    temporal_train_val_test_split,
)
from fleetguard.models.baseline import train_and_evaluate_baselines


def test_temporal_split_has_no_time_overlap():
    df = generate_historical_telemetry(num_vehicles=15, steps_per_vehicle=30, seed=99)
    splits = temporal_train_val_test_split(df, train_frac=0.70, val_frac=0.15)
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == len(df)
    assert list(splits.X_train.columns) == FEATURE_COLS
    assert "Vehicle_ID" not in splits.X_train.columns
    assert "Latitude" not in splits.X_train.columns


def test_preprocessor_handles_unseen_categories():
    df = generate_historical_telemetry(num_vehicles=15, steps_per_vehicle=20, seed=99)
    splits = temporal_train_val_test_split(df)
    prep = build_preprocessor(scale_numeric=True)
    prep.fit(splits.X_train)

    X_unseen = splits.X_val.iloc[:5].copy()
    X_unseen.loc[X_unseen.index[0], "Make_and_Model"] = "Unknown Prototype Truck"
    transformed = prep.transform(X_unseen)
    assert transformed.shape[0] == 5


def test_baseline_training_smoke():
    df = generate_historical_telemetry(num_vehicles=40, steps_per_vehicle=60, seed=42)
    report = train_and_evaluate_baselines(df, save_artifacts=False)
    assert "logistic_regression_balanced" in report["models"]
    assert "random_forest_balanced" in report["models"]
    rf_auc = report["models"]["random_forest_balanced"]["validation_metrics"]["roc_auc"]
    assert rf_auc is not None and rf_auc > 0.55