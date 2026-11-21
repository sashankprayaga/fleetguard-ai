from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

ALLOWED_DATA_SOURCES = {"synthetic_historical", "synthetic_live_stream", "uploaded_batch"}


class TelemetryRecordSchema(BaseModel):
    """Row-level contract for validating incoming fleet telemetry packets."""

    Timestamp: datetime
    Vehicle_ID: str = Field(min_length=3, max_length=32)
    Make_and_Model: str
    Vehicle_Type: str
    Year_of_Manufacture: int = Field(ge=2000, le=2030)
    Route_Corridor: str
    Route_Progress: float = Field(ge=0.0, le=1.0)
    Latitude: float = Field(ge=47.0, le=55.5)
    Longitude: float = Field(ge=5.5, le=15.5)
    Speed_kmh: float = Field(ge=0.0, le=140.0)
    Usage_Hours: float = Field(ge=0.0)
    Load_Capacity: float = Field(gt=0.0)
    Actual_Load: float = Field(ge=0.0)
    Engine_Temperature: float = Field(ge=-20.0, le=160.0)
    Tire_Pressure: float = Field(ge=20.0, le=150.0)
    Fuel_Consumption: float = Field(ge=0.0, le=120.0)
    Battery_Status: float = Field(ge=0.0, le=100.0)
    Vibration_Levels: float = Field(ge=0.0, le=25.0)
    Oil_Quality: float = Field(ge=0.0, le=100.0)
    Failure_History: int = Field(ge=0)
    Days_Since_Last_Maintenance: float = Field(ge=0.0)
    Last_Maintenance_Type: str
    Brake_Condition: Literal["Good", "Fair", "Worn"]
    PCR: float = Field(ge=0.0, le=2.0)
    UIR: float = Field(ge=0.0)
    TPI: float = Field(ge=0.0)
    Data_Source: str


@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    column_count: int
    missing_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    invalid_sources: list[str] = field(default_factory=list)
    sample_row_errors: list[str] = field(default_factory=list)


def validate_telemetry_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
    sample_size: int = 200,
) -> ValidationReport:
    """
    Validate a telemetry DataFrame using fast vectorized column/null checks
    plus Pydantic schema verification on a stratified sample of rows.
    """
    required_cols = list(TelemetryRecordSchema.model_fields.keys())
    if require_target:
        required_cols.append("Maintenance_Required")

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return ValidationReport(
            is_valid=False,
            row_count=len(df),
            column_count=len(df.columns),
            missing_columns=missing_cols,
        )

    null_series = df[required_cols].isna().sum()
    null_counts = {col: int(cnt) for col, cnt in null_series.items() if cnt > 0}

    invalid_sources = sorted(set(df["Data_Source"].unique()) - ALLOWED_DATA_SOURCES)

    target_invalid = False
    if require_target:
        valid_targets = set(df["Maintenance_Required"].dropna().unique()).issubset({0, 1})
        if not valid_targets:
            target_invalid = True

    # Pydantic validation on a sample of rows for fast, detailed type/bound checks
    sample_df = df.head(sample_size) if len(df) <= sample_size else df.sample(n=sample_size, random_state=42)
    row_errors: list[str] = []
    if target_invalid:
        row_errors.append("Maintenance_Required contains values outside {0, 1}.")

    for idx, record in enumerate(sample_df.to_dict(orient="records")):
        try:
            TelemetryRecordSchema(**record)
        except ValidationError as exc:
            row_errors.append(f"Row {idx}: {exc.errors()[0]['loc']} -> {exc.errors()[0]['msg']}")
            if len(row_errors) >= 5:
                break

    is_valid = (
        len(missing_cols) == 0
        and len(null_counts) == 0
        and len(invalid_sources) == 0
        and len(row_errors) == 0
    )

    return ValidationReport(
        is_valid=is_valid,
        row_count=len(df),
        column_count=len(df.columns),
        missing_columns=missing_cols,
        null_counts=null_counts,
        invalid_sources=invalid_sources,
        sample_row_errors=row_errors,
    )