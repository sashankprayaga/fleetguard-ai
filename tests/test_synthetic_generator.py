from fleetguard.data.synthetic_generator import (
    ROUTE_CORRIDORS,
    VEHICLE_SPECS,
    create_fleet_registry,
    generate_historical_telemetry,
)


def test_fleet_diversity_and_german_corridors():
    fleet = create_fleet_registry(num_vehicles=200, seed=42)
    expected_types = {spec[1] for spec in VEHICLE_SPECS}
    assert set(fleet["Vehicle_Type"].unique()) == expected_types
    assert set(fleet["Route_Corridor"].unique()) == set(ROUTE_CORRIDORS.keys())


def test_historical_telemetry_schema_and_german_gps():
    df = generate_historical_telemetry(num_vehicles=25, steps_per_vehicle=20, seed=123)
    assert df.shape[0] == 500
    assert df["Vehicle_ID"].nunique() == 25
    assert (df["Data_Source"] == "synthetic_historical").all()
    assert set(df["Maintenance_Required"].unique()).issubset({0, 1})
    assert df["Latitude"].between(47.2, 55.1).all()
    assert df["Longitude"].between(5.8, 15.1).all()
    assert df.isna().sum().sum() == 0