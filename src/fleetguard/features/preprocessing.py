from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fleetguard.data.eda import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL

FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TemporalSplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_end_ts: str
    val_end_ts: str


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TemporalSplitData:
    """
    Split telemetry chronologically by Timestamp so future observations never leak
    into training or validation folds.
    """
    ordered = df.sort_values(["Timestamp", "Vehicle_ID"]).reset_index(drop=True)
    unique_ts = ordered["Timestamp"].drop_duplicates().sort_values().reset_index(drop=True)

    n_ts = len(unique_ts)
    train_cutoff_ts = unique_ts.iloc[int(n_ts * train_frac) - 1]
    val_cutoff_ts = unique_ts.iloc[int(n_ts * (train_frac + val_frac)) - 1]

    train_mask = ordered["Timestamp"] <= train_cutoff_ts
    val_mask = (ordered["Timestamp"] > train_cutoff_ts) & (ordered["Timestamp"] <= val_cutoff_ts)
    test_mask = ordered["Timestamp"] > val_cutoff_ts

    train_df = ordered.loc[train_mask]
    val_df = ordered.loc[val_mask]
    test_df = ordered.loc[test_mask]

    return TemporalSplitData(
        X_train=train_df[FEATURE_COLS].copy(),
        y_train=train_df[TARGET_COL].astype(int).copy(),
        X_val=val_df[FEATURE_COLS].copy(),
        y_val=val_df[TARGET_COL].astype(int).copy(),
        X_test=test_df[FEATURE_COLS].copy(),
        y_test=test_df[TARGET_COL].astype(int).copy(),
        train_end_ts=str(train_cutoff_ts),
        val_end_ts=str(val_cutoff_ts),
    )


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Create a leak-free Scikit-learn ColumnTransformer for numerical and categorical features.
    """
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

def build_preprocessor_for_features(
    num_features: list[str] | None = None,
    cat_features: list[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a ColumnTransformer for any explicit numerical + categorical feature lists."""
    nums = num_features if num_features is not None else NUMERICAL_FEATURES
    cats = cat_features if cat_features is not None else CATEGORICAL_FEATURES

    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=num_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, nums),
            ("cat", categorical_pipeline, cats),
        ],
        remainder="drop",
    )