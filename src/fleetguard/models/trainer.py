from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fleetguard.config.settings import PATHS
from fleetguard.data.validator import validate_telemetry_dataframe
from fleetguard.features.engineering import (
    ALL_MODEL_CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES,
    ALL_MODEL_NUMERICAL_FEATURES,
    HORIZON_TARGET_COL,
    add_domain_and_rolling_features,
    add_forward_failure_horizon,
)
from fleetguard.features.preprocessing import build_preprocessor_for_features

INSPECTION_COST_EUR = 350.0
BREAKDOWN_COST_EUR = 4500.0
HORIZON_STEPS = 18  # 18 steps * 4h = 72-hour early-warning window


def find_cost_optimal_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    inspection_cost: float = INSPECTION_COST_EUR,
    breakdown_cost: float = BREAKDOWN_COST_EUR,
    max_alert_rate: float = 0.12,
) -> dict:
    """
    Sweep probability thresholds on the validation set to minimize total expected
    fleet maintenance + breakdown cost in EUR, subject to workshop bay capacity
    (alert rate <= max_alert_rate).
    """
    best_thresh = 0.50
    best_total_cost = float("inf")
    baseline_no_model_cost = float(y_true.sum() * breakdown_cost)

    for thresh in np.arange(0.05, 0.91, 0.01):
        preds = (y_prob >= thresh).astype(int)
        if preds.mean() > max_alert_rate:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        total_cost = (tp + fp) * inspection_cost + fn * breakdown_cost
        if total_cost < best_total_cost:
            best_total_cost = float(total_cost)
            best_thresh = round(float(thresh), 2)

    # Fallback if even 0.90 exceeds max_alert_rate
    if best_total_cost == float("inf"):
        best_thresh = float(np.quantile(y_prob, 1.0 - max_alert_rate))
        preds = (y_prob >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        best_total_cost = float((tp + fp) * inspection_cost + fn * breakdown_cost)
        best_thresh = round(best_thresh, 2)

    net_savings = baseline_no_model_cost - best_total_cost
    return {
        "optimal_threshold": best_thresh,
        "max_workshop_capacity_pct": round(max_alert_rate * 100, 1),
        "inspection_cost_eur": inspection_cost,
        "breakdown_cost_eur": breakdown_cost,
        "validation_total_cost_eur": round(best_total_cost, 2),
        "validation_no_model_cost_eur": round(baseline_no_model_cost, 2),
        "validation_net_savings_eur": round(net_savings, 2),
    }


def evaluate_at_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    has_both = y_true.nunique() >= 2

    total_cost = (tp + fp) * INSPECTION_COST_EUR + fn * BREAKDOWN_COST_EUR
    no_model_cost = float(y_true.sum() * BREAKDOWN_COST_EUR)

    return {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if has_both else None,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4) if has_both else None,
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "alert_rate_pct": round(float(preds.mean() * 100), 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_cost_eur": round(float(total_cost), 2),
        "net_savings_vs_reactive_eur": round(float(no_model_cost - total_cost), 2),
    }


def train_champion_models(
    df: pd.DataFrame,
    horizon_steps: int = HORIZON_STEPS,
    save_artifacts: bool = True,
) -> dict:
    """
    Train 72-hour forward-horizon predictive maintenance models (XGBoost & LightGBM)
    with strict temporal embargo gaps between splits to prevent future target leakage.
    """
    validation = validate_telemetry_dataframe(df, require_target=True)
    if not validation.is_valid:
        raise ValueError(f"Telemetry validation failed: {validation.sample_row_errors}")

    enriched = add_domain_and_rolling_features(df)
    enriched = add_forward_failure_horizon(enriched, horizon_steps=horizon_steps)
    enriched = enriched.dropna(subset=[HORIZON_TARGET_COL]).reset_index(drop=True)

    unique_ts = enriched["Timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    n_ts = len(unique_ts)
    train_idx = int(n_ts * 0.70) - 1
    val_idx = int(n_ts * 0.85) - 1

    # Purge `horizon_steps` timestamps before each cutoff so forward labels never cross splits
    train_cutoff = unique_ts.iloc[max(0, train_idx - horizon_steps)]
    val_start = unique_ts.iloc[train_idx + 1]
    val_cutoff = unique_ts.iloc[max(train_idx + 2, val_idx - horizon_steps)]
    test_start = unique_ts.iloc[val_idx + 1]

    train_df = enriched[enriched["Timestamp"] <= train_cutoff]
    val_df = enriched[(enriched["Timestamp"] >= val_start) & (enriched["Timestamp"] <= val_cutoff)]
    test_df = enriched[enriched["Timestamp"] >= test_start]

    X_train, y_train = train_df[ALL_MODEL_FEATURES], train_df[HORIZON_TARGET_COL].astype(int)
    X_val, y_val = val_df[ALL_MODEL_FEATURES], val_df[HORIZON_TARGET_COL].astype(int)
    X_test, y_test = test_df[ALL_MODEL_FEATURES], test_df[HORIZON_TARGET_COL].astype(int)

    candidates: dict[str, Pipeline] = {
        "xgboost": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor_for_features(
                        ALL_MODEL_NUMERICAL_FEATURES, ALL_MODEL_CATEGORICAL_FEATURES, scale_numeric=False
                    ),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        eval_metric="aucpr",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor_for_features(
                        ALL_MODEL_NUMERICAL_FEATURES, ALL_MODEL_CATEGORICAL_FEATURES, scale_numeric=False
                    ),
                ),
                (
                    "classifier",
                    LGBMClassifier(
                        n_estimators=250,
                        num_leaves=31,
                        learning_rate=0.05,
                        subsample_for_bin=50000,
                        colsample_bytree=0.85,
                        subsample=0.85,
                        reg_lambda=2.0,
                        n_jobs=-1,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }

    candidate_reports: dict[str, dict] = {}
    best_name = ""
    best_pr_auc = -1.0

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]
        cost_opt = find_cost_optimal_threshold(y_val, val_probs)
        val_eval = evaluate_at_threshold(y_val, val_probs, threshold=cost_opt["optimal_threshold"])

        candidate_reports[name] = {
            "cost_optimization": cost_opt,
            "validation_metrics": val_eval,
        }
        if (val_eval["pr_auc"] or 0.0) > best_pr_auc:
            best_pr_auc = val_eval["pr_auc"] or 0.0
            best_name = name

    champion_pipe = candidates[best_name]
    champion_threshold = candidate_reports[best_name]["cost_optimization"]["optimal_threshold"]

    test_probs = champion_pipe.predict_proba(X_test)[:, 1]
    test_eval = evaluate_at_threshold(y_test, test_probs, threshold=champion_threshold)

    summary = {
        "prediction_horizon_hours": horizon_steps * 4,
        "target_column": HORIZON_TARGET_COL,
        "train_positive_rate_pct": round(float(y_train.mean() * 100), 2),
        "test_positive_rate_pct": round(float(y_test.mean() * 100), 2),
        "champion_model": best_name,
        "optimal_threshold": champion_threshold,
        "feature_count": len(ALL_MODEL_FEATURES),
        "feature_columns": ALL_MODEL_FEATURES,
        "candidates": candidate_reports,
        "champion_heldout_test_metrics": test_eval,
    }

    if save_artifacts:
        PATHS.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion_pipe, PATHS.models_dir / "champion_model.joblib")
        with open(PATHS.models_dir / "champion_metadata.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    raw_path = PATHS.raw_data / "synthetic_fleet_history.parquet"
    df = pd.read_parquet(raw_path)
    report = train_champion_models(df, save_artifacts=True)
    print(json.dumps(report, indent=2))