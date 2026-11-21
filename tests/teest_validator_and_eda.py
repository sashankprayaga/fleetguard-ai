from fleetguard.data.eda import run_exploratory_audit
from fleetguard.data.synthetic_generator import generate_historical_telemetry
from fleetguard.data.validator import validate_telemetry_dataframe


def test_validator_accepts_clean_data():
    df = generate_historical_telemetry(num_vehicles=5, steps_per_vehicle=10, seed=7)
    report = validate_telemetry_dataframe(df, require_target=True)
    assert report.is_valid is True
    assert report.missing_columns == []
    assert report.null_counts == {}


def test_validator_catches_out_of_bounds_sensor():
    df = generate_historical_telemetry(num_vehicles=5, steps_per_vehicle=10, seed=7)
    df.loc[0, "Oil_Quality"] = 145.0  # Impossible (> 100%)
    report = validate_telemetry_dataframe(df, require_target=True)
    assert report.is_valid is False
    assert len(report.sample_row_errors) > 0


def test_eda_accounts_for_all_columns():
    df = generate_historical_telemetry(num_vehicles=10, steps_per_vehicle=15, seed=7)
    audit = run_exploratory_audit(df)
    assert audit["validation_passed"] is True
    assert audit["feature_governance"]["unaccounted_columns"] == []