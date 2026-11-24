from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
import pandas as pd

from fleetguard.config.settings import PATHS


def get_connection() -> sqlite3.Connection:
    PATHS.database_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PATHS.sqlite_db)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create SQLite tables for scored telemetry snapshots, alerts, and drift audits."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fleet_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scored_at TEXT NOT NULL,
                data_source TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                make_and_model TEXT,
                vehicle_type TEXT,
                route_corridor TEXT,
                latitude REAL,
                longitude REAL,
                speed_kmh REAL,
                engine_temperature REAL,
                oil_quality REAL,
                vibration_levels REAL,
                days_since_last_maintenance REAL,
                brake_condition TEXT,
                failure_prob_72h REAL NOT NULL,
                alert_triggered INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                top_drivers_json TEXT
            );

            CREATE TABLE IF NOT EXISTS monitoring_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                data_source TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                alert_count INTEGER NOT NULL,
                alert_rate_pct REAL NOT NULL,
                avg_failure_prob REAL NOT NULL,
                drift_status TEXT NOT NULL,
                max_psi REAL NOT NULL,
                feature_psi_json TEXT NOT NULL,
                mlflow_run_id TEXT
            );
            """
        )


def save_scored_batch(scored_df: pd.DataFrame, monitoring_summary: dict) -> None:
    """Persist scored fleet rows and batch monitoring summary into SQLite."""
    init_database()
    now_iso = datetime.now(timezone.utc).isoformat()

    records = []
    for _, row in scored_df.iterrows():
        records.append(
            (
                now_iso,
                str(row["Data_Source"]),
                str(row["Vehicle_ID"]),
                str(row["Timestamp"]),
                str(row["Make_and_Model"]),
                str(row["Vehicle_Type"]),
                str(row["Route_Corridor"]),
                float(row["Latitude"]),
                float(row["Longitude"]),
                float(row["Speed_kmh"]),
                float(row["Engine_Temperature"]),
                float(row["Oil_Quality"]),
                float(row["Vibration_Levels"]),
                float(row["Days_Since_Last_Maintenance"]),
                str(row["Brake_Condition"]),
                float(row["Failure_Prob_72h"]),
                int(row["Alert_Triggered"]),
                str(row["Risk_Tier"]),
                str(row.get("Top_Drivers_JSON", "[]")),
            )
        )

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO fleet_predictions (
                scored_at, data_source, vehicle_id, timestamp, make_and_model,
                vehicle_type, route_corridor, latitude, longitude, speed_kmh,
                engine_temperature, oil_quality, vibration_levels,
                days_since_last_maintenance, brake_condition, failure_prob_72h,
                alert_triggered, risk_tier, top_drivers_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        conn.execute(
            """
            INSERT INTO monitoring_logs (
                logged_at, data_source, batch_size, alert_count, alert_rate_pct,
                avg_failure_prob, drift_status, max_psi, feature_psi_json, mlflow_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                str(monitoring_summary["data_source"]),
                int(monitoring_summary["batch_size"]),
                int(monitoring_summary["alert_count"]),
                float(monitoring_summary["alert_rate_pct"]),
                float(monitoring_summary["avg_failure_prob"]),
                str(monitoring_summary["drift_status"]),
                float(monitoring_summary["max_psi"]),
                json.dumps(monitoring_summary["feature_psi"]),
                str(monitoring_summary.get("mlflow_run_id", "")),
            ),
        )