from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from fleetguard.config.settings import PATHS

# (Make_and_Model, Vehicle_Type, Load_Capacity_kg, Base_Temp_C, Temp_Load_Factor)
VEHICLE_SPECS = [
    ("Volvo FH16 750", "Heavy Haul Truck", 28000.0, 84.0, 17.0),
    ("Scania R500 V8", "Heavy Haul Truck", 26000.0, 83.0, 16.5),
    ("Mercedes-Benz Actros 1851", "Heavy Haul Truck", 25000.0, 82.5, 16.0),
    ("MAN TGX 18.510", "Heavy Haul Truck", 25500.0, 83.0, 16.0),
    ("DAF XG+ 530", "Heavy Haul Truck", 25000.0, 82.0, 15.5),
    ("DAF XF 480 Reefer", "Refrigerated Truck", 20000.0, 83.5, 16.5),
    ("Scania P320 ColdChain", "Refrigerated Truck", 18000.0, 82.5, 15.5),
    ("Freightliner eCascadia", "Electric Semi", 22000.0, 66.0, 11.0),
    ("Volvo FH Electric", "Electric Semi", 24000.0, 65.0, 11.5),
    ("Mercedes-Benz eActros 600", "Electric Semi", 22000.0, 64.5, 11.0),
    ("Mercedes-Benz Arocs Tanker", "Tanker Truck", 27000.0, 84.5, 17.5),
    ("Volvo FMX ADR Tanker", "Tanker Truck", 26500.0, 84.0, 17.0),
    ("Iveco Eurocargo 120E", "Regional Box Truck", 12000.0, 80.0, 13.5),
    ("MAN TGL 12.250 Flatbed", "Flatbed Carrier", 14000.0, 81.0, 14.5),
]

# Multi-waypoint European logistics corridors: list of (lat, lon) highway checkpoints
# Multi-waypoint German Autobahn logistics corridors: list of (lat, lon) highway checkpoints
ROUTE_CORRIDORS: dict[str, list[tuple[float, float]]] = {
    "A3 Rhine-Ruhr -> Frankfurt": [
        (51.1714, 6.9395),   # Hilden / Düsseldorf
        (50.9375, 6.9603),   # Cologne
        (50.4350, 7.8260),   # Montabaur
        (50.1109, 8.6821),   # Frankfurt am Main
    ],
    "A1 Ruhr -> Hamburg": [
        (51.5136, 7.4653),   # Dortmund
        (51.9607, 7.6261),   # Münster
        (53.0793, 8.8017),   # Bremen
        (53.5511, 9.9937),   # Hamburg
    ],
    "A2 Hanover -> Berlin": [
        (52.3759, 9.7320),   # Hanover
        (52.1205, 11.6276),  # Magdeburg
        (52.4009, 12.5597),  # Brandenburg
        (52.5200, 13.4050),  # Berlin
    ],
    "A9 Nuremberg -> Munich": [
        (49.4521, 11.0767),  # Nuremberg
        (48.7665, 11.4257),  # Ingolstadt
        (48.1351, 11.5820),  # Munich
    ],
    "A5 Frankfurt -> Freiburg": [
        (50.1109, 8.6821),   # Frankfurt am Main
        (49.4875, 8.4660),   # Mannheim
        (49.0069, 8.4037),   # Karlsruhe
        (47.9990, 7.8421),   # Freiburg im Breisgau
    ],
    "A8 Stuttgart -> Munich": [
        (48.7758, 9.1829),   # Stuttgart
        (48.4011, 9.9876),   # Ulm
        (48.3705, 10.8978),  # Augsburg
        (48.1351, 11.5820),  # Munich
    ],
    "A7 Hamburg -> Kassel": [
        (53.5511, 9.9937),   # Hamburg
        (52.3759, 9.7320),   # Hanover
        (51.5413, 9.9158),   # Göttingen
        (51.3127, 9.4797),   # Kassel
    ],
    "A7 Kassel -> Ulm": [
        (51.3127, 9.4797),   # Kassel
        (50.5558, 9.6808),   # Fulda
        (49.7913, 9.9534),   # Würzburg
        (48.4011, 9.9876),   # Ulm
    ],
    "A4 Cologne -> Dresden": [
        (50.9375, 6.9603),   # Cologne
        (50.9848, 11.0299),  # Erfurt
        (50.8278, 12.9214),  # Chemnitz
        (51.0504, 13.7373),  # Dresden
    ],
    "A9 Berlin -> Nuremberg": [
        (52.5200, 13.4050),  # Berlin
        (51.3397, 12.3731),  # Leipzig
        (50.3135, 11.9128),  # Hof
        (49.4521, 11.0767),  # Nuremberg
    ],
    "A6 Mannheim -> Nuremberg": [
        (49.4875, 8.4660),   # Mannheim
        (49.1427, 9.2109),   # Heilbronn
        (49.3006, 10.5719),  # Ansbach
        (49.4521, 11.0767),  # Nuremberg
    ],
    "A24 Hamburg -> Berlin": [
        (53.5511, 9.9937),   # Hamburg
        (53.6355, 11.4012),  # Schwerin
        (53.1628, 12.4864),  # Wittstock
        (52.5200, 13.4050),  # Berlin
    ],
}

MAINTENANCE_TYPES = [
    "Oil Check",
    "Brake Service",
    "Tire Replacement",
    "Engine Overhaul",
    "Cooling System Flush",
    "Battery & HV Diagnostics",
]


def _interpolate_gps(
    corridor: str, progress: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear interpolation across multi-waypoint highway corridors."""
    waypoints = np.array(ROUTE_CORRIDORS[corridor], dtype=float)
    num_segments = len(waypoints) - 1

    scaled = np.clip(progress, 0.0, 0.999999) * num_segments
    seg_idx = np.floor(scaled).astype(int)
    seg_t = scaled - seg_idx

    start_pts = waypoints[seg_idx]
    end_pts = waypoints[seg_idx + 1]

    lat = start_pts[:, 0] + (end_pts[:, 0] - start_pts[:, 0]) * seg_t + rng.normal(0, 0.003, size=len(progress))
    lon = start_pts[:, 1] + (end_pts[:, 1] - start_pts[:, 1]) * seg_t + rng.normal(0, 0.003, size=len(progress))
    return np.round(lat, 5), np.round(lon, 5)


def create_fleet_registry(num_vehicles: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create static master metadata and assigned corridor for each truck."""
    rng = np.random.default_rng(seed)
    spec_indices = rng.integers(0, len(VEHICLE_SPECS), size=num_vehicles)
    corridor_names = list(ROUTE_CORRIDORS.keys())
    corridor_indices = rng.integers(0, len(corridor_names), size=num_vehicles)

    records = []
    for idx in range(num_vehicles):
        make_model, v_type, capacity, base_temp, temp_load_factor = VEHICLE_SPECS[spec_indices[idx]]
        records.append(
            {
                "Vehicle_ID": f"TRK-{1001 + idx:04d}",
                "Make_and_Model": make_model,
                "Vehicle_Type": v_type,
                "Year_of_Manufacture": int(rng.integers(2016, 2026)),
                "Load_Capacity": capacity,
                "Base_Temp_C": base_temp,
                "Temp_Load_Factor": temp_load_factor,
                "Base_Usage_Hours": float(rng.uniform(500.0, 14000.0)),
                "Prior_Failures": int(rng.poisson(0.6)),
                "Route_Corridor": corridor_names[corridor_indices[idx]],
                "Initial_Progress": float(rng.uniform(0.0, 1.0)),
            }
        )
    return pd.DataFrame(records)


def generate_historical_telemetry(
    num_vehicles: int = 500,
    steps_per_vehicle: int = 500,
    step_hours: int = 4,
    seed: int = 42,
    data_source: str = "synthetic_historical",
) -> pd.DataFrame:
    """Generate time-ordered fleet telemetry with powertrain-aware wear and GPS movement."""
    rng = np.random.default_rng(seed)
    fleet = create_fleet_registry(num_vehicles=num_vehicles, seed=seed)

    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    all_frames: list[pd.DataFrame] = []

    for row in fleet.itertuples(index=False):
        n = steps_per_vehicle
        timestamps = [start_time + timedelta(hours=i * step_hours) for i in range(n)]

        # Kinematics & multi-waypoint GPS movement along assigned corridor
        speed_kmh = np.clip(rng.normal(78.0, 10.0, size=n), 35.0, 95.0)
        raw_travel = row.Initial_Progress + np.cumsum((speed_kmh * step_hours) / 520.0)
        route_progress = 1.0 - np.abs((raw_travel % 2.0) - 1.0)
        latitude, longitude = _interpolate_gps(row.Route_Corridor, route_progress, rng)

        # Cumulative operating hours and payload
        delta_hours = rng.uniform(0.5, float(step_hours), size=n)
        usage_hours = row.Base_Usage_Hours + np.cumsum(delta_hours)
        actual_load = rng.uniform(0.25, 1.05, size=n) * row.Load_Capacity
        pcr = actual_load / row.Load_Capacity  # Proportional Capacity Ratio

        # Periodic scheduled maintenance resets
        days_since_maint = np.zeros(n, dtype=float)
        current_days = float(rng.uniform(0, 60))
        last_maint_type = rng.choice(MAINTENANCE_TYPES)
        maint_types = []

        for i in range(n):
            current_days += step_hours / 24.0
            if current_days > rng.uniform(110.0, 150.0):
                current_days = 0.0
                last_maint_type = rng.choice(MAINTENANCE_TYPES)
            days_since_maint[i] = round(current_days, 2)
            maint_types.append(last_maint_type)

        vehicle_age = max(1, 2026 - row.Year_of_Manufacture)
        uir = usage_hours / (vehicle_age * 2000.0)  # Use Intensity Ratio
        is_electric = row.Vehicle_Type == "Electric Semi"

        oil_quality = np.clip(
            100.0 - (days_since_maint * (0.20 if is_electric else 0.42)) - (pcr * 7.0) + rng.normal(0, 3.0, size=n),
            10.0,
            100.0,
        )
        engine_temp = np.clip(
            row.Base_Temp_C
            + (pcr * row.Temp_Load_Factor)
            + ((100.0 - oil_quality) * 0.14)
            + rng.normal(0, 2.5, size=n),
            55.0,
            125.0,
        )
        tire_pressure = np.clip(
            105.0 - (pcr * 4.5) - (days_since_maint * 0.05) + rng.normal(0, 2.0, size=n),
            75.0,
            115.0,
        )
        vibration = np.clip(
            (0.9 if is_electric else 1.2)
            + (uir * 0.6)
            + ((100.0 - oil_quality) * 0.025)
            + (pcr * 0.9)
            + rng.normal(0, 0.25, size=n),
            0.4,
            8.5,
        )
        battery_status = np.clip(
            98.0
            - (vehicle_age * 1.8)
            - (days_since_maint * (0.11 if is_electric else 0.06))
            - (pcr * (4.0 if is_electric else 1.0))
            + rng.normal(0, 2.0, size=n),
            40.0,
            100.0,
        )
        # Fuel/energy consumption equivalent (L/100km or kWh-eq/100km scaled)
        fuel_consumption = np.clip(
            (18.0 if is_electric else 24.0)
            + (pcr * 14.0)
            + ((100.0 - oil_quality) * 0.05)
            + rng.normal(0, 1.5, size=n),
            10.0,
            55.0,
        )
        tpi = (engine_temp * tire_pressure) / 100.0  # Temperature Pressure Index

        brake_condition = np.where(
            (days_since_maint > 95) | (vibration > 4.2),
            "Worn",
            np.where(days_since_maint > 45, "Fair", "Good"),
        )

        # Powertrain-relative thermal excess so Electric Semis are judged against their own baseline
        thermal_excess = np.maximum(0.0, engine_temp - (row.Base_Temp_C + 12.0))
        battery_deficit = np.maximum(0.0, (75.0 if is_electric else 60.0) - battery_status)

        risk_logit = (
            -5.2
            + 0.068 * thermal_excess
            + 0.045 * np.maximum(0.0, 65.0 - oil_quality)
            + 0.55 * np.maximum(0.0, vibration - 3.2)
            + 0.018 * np.maximum(0.0, days_since_maint - 75.0)
            + 0.04 * battery_deficit
            + 0.25 * row.Prior_Failures
            + rng.normal(0, 0.45, size=n)
        )
        failure_prob = 1.0 / (1.0 + np.exp(-risk_logit))
        maintenance_required = (rng.uniform(0, 1, size=n) < failure_prob).astype(int)

        df_v = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Vehicle_ID": row.Vehicle_ID,
                "Make_and_Model": row.Make_and_Model,
                "Vehicle_Type": row.Vehicle_Type,
                "Year_of_Manufacture": row.Year_of_Manufacture,
                "Route_Corridor": row.Route_Corridor,
                "Route_Progress": np.round(route_progress, 4),
                "Latitude": latitude,
                "Longitude": longitude,
                "Speed_kmh": np.round(speed_kmh, 1),
                "Usage_Hours": np.round(usage_hours, 2),
                "Load_Capacity": row.Load_Capacity,
                "Actual_Load": np.round(actual_load, 1),
                "Engine_Temperature": np.round(engine_temp, 2),
                "Tire_Pressure": np.round(tire_pressure, 2),
                "Fuel_Consumption": np.round(fuel_consumption, 2),
                "Battery_Status": np.round(battery_status, 2),
                "Vibration_Levels": np.round(vibration, 3),
                "Oil_Quality": np.round(oil_quality, 2),
                "Failure_History": row.Prior_Failures,
                "Days_Since_Last_Maintenance": days_since_maint,
                "Last_Maintenance_Type": maint_types,
                "Brake_Condition": brake_condition,
                "PCR": np.round(pcr, 4),
                "UIR": np.round(uir, 4),
                "TPI": np.round(tpi, 2),
                "Maintenance_Required": maintenance_required,
                "Data_Source": data_source,
            }
        )
        all_frames.append(df_v)

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.sort_values(["Timestamp", "Vehicle_ID"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    return full_df


def build_and_save_dataset(output_path: Path | None = None) -> Path:
    """Generate and persist the 250,000-row historical dataset + fleet registry."""
    PATHS.raw_data.mkdir(parents=True, exist_ok=True)
    target_path = output_path or (PATHS.raw_data / "synthetic_fleet_history.parquet")
    registry_path = PATHS.raw_data / "fleet_registry.parquet"

    fleet_df = create_fleet_registry(num_vehicles=500, seed=42)
    fleet_df.to_parquet(registry_path, index=False)

    history_df = generate_historical_telemetry(num_vehicles=500, steps_per_vehicle=500, seed=42)
    history_df.to_parquet(target_path, index=False)
    return target_path


if __name__ == "__main__":
    out = build_and_save_dataset()
    print(f"Saved synthetic fleet dataset to: {out}")