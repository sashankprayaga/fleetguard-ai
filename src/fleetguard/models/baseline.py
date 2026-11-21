from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from fleetguard.config.settings import PATHS
from fleetguard.data.validator import validate_telemetry_dataframe
from fleetguard.features.preprocessing import (
    build_preprocessor,
    temporal_train_val_test_split,
)


def evaluate_binary_classifier(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.50
) -> dict[str, float | None]:
    """Compute imbalance-aware classification metrics at a given probability threshold."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    has_both_classes = y.nunique() >= 2

    return {
        "roc_auc": round(float(roc_auc_score(y, probs)), 4) if has_both_classes else None,
        "pr_auc": round(float(average_precision_score(y, probs)), 4) if has_both_classes else None,
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "predicted_positive_rate_pct": round(float(preds.mean() * 100), 2),
    }


def train_and_evaluate_baselines(df: pd.DataFrame, save_artifacts: bool = True) -> dict:
    """Validate dataset, create temporal splits, and train baseline models."""
    validation = validate_telemetry_dataframe(df, require_target=True)
    if not validation.is_valid:
        raise ValueError(f"Telemetry validation failed: {validation.sample_row_errors}")

    splits = temporal_train_val_test_split(df)

    candidates: dict[str, Pipeline] = {
        "logistic_regression_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results: dict[str, dict] = {}
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)

    for name, pipeline in candidates.items():
        pipeline.fit(splits.X_train, splits.y_train)
        val_metrics = evaluate_binary_classifier(pipeline, splits.X_val, splits.y_val)
        results[name] = {
            "validation_metrics": val_metrics,
        }
        if save_artifacts:
            joblib.dump(pipeline, PATHS.models_dir / f"{name}.joblib")

    summary = {
        "split_info": {
            "train_rows": int(len(splits.X_train)),
            "val_rows": int(len(splits.X_val)),
            "test_rows": int(len(splits.X_test)),
            "train_end_ts": splits.train_end_ts,
            "val_end_ts": splits.val_end_ts,
            "train_positive_rate_pct": round(float(splits.y_train.mean() * 100), 2),
            "val_positive_rate_pct": round(float(splits.y_val.mean() * 100), 2),
            "test_positive_rate_pct": round(float(splits.y_test.mean() * 100), 2),
        },
        "models": results,
    }

    if save_artifacts:
        with open(PATHS.models_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    raw_path = PATHS.raw_data / "synthetic_fleet_history.parquet"
    df = pd.read_parquet(raw_path)
    report = train_and_evaluate_baselines(df, save_artifacts=True)
    print(json.dumps(report, indent=2))