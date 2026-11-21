from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from fleetguard.config.settings import PATHS
from fleetguard.features.engineering import (
    ALL_MODEL_CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES,
    ALL_MODEL_NUMERICAL_FEATURES,
    add_domain_and_rolling_features,
)


def _map_transformed_to_original_feature(transformed_name: str) -> str:
    """Map ColumnTransformer output feature names back to original domain feature names."""
    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")
    if transformed_name.startswith("cat__"):
        raw = transformed_name.removeprefix("cat__")
        for cat_col in ALL_MODEL_CATEGORICAL_FEATURES:
            if raw.startswith(f"{cat_col}_"):
                return cat_col
        return raw
    return transformed_name


class FleetShapExplainer:
    """Wrapper around TreeExplainer for the FleetGuard champion pipeline."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())
        self.original_feature_map = [
            _map_transformed_to_original_feature(name) for name in self.transformed_feature_names
        ]
        self.explainer = shap.TreeExplainer(self.classifier)

    @classmethod
    def from_saved_champion(cls, model_path: Path | None = None) -> "FleetShapExplainer":
        target = model_path or (PATHS.models_dir / "champion_model.joblib")
        pipeline = joblib.load(target)
        return cls(pipeline)

    def _aggregate_shap_by_original_feature(self, shap_matrix: np.ndarray) -> pd.DataFrame:
        """Sum SHAP contributions across one-hot columns belonging to the same original feature."""
        raw_df = pd.DataFrame(shap_matrix, columns=self.original_feature_map)
        # Group identical original column names and sum their SHAP attributions
        aggregated = raw_df.T.groupby(level=0).sum().T
        # Ensure consistent column ordering across all 27 model features
        ordered_cols = [c for c in ALL_MODEL_FEATURES if c in aggregated.columns]
        return aggregated[ordered_cols]

    def compute_shap_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of per-row, per-feature SHAP contributions (log-odds impact)."""
        X_model = X[ALL_MODEL_FEATURES]
        X_trans = self.preprocessor.transform(X_model)
        shap_vals = self.explainer.shap_values(X_trans)
        # LightGBM binary classifiers sometimes return a list [class_0, class_1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return self._aggregate_shap_by_original_feature(np.asarray(shap_vals))

    def compute_global_importance(
        self, df: pd.DataFrame, sample_size: int = 2000, seed: int = 42
    ) -> list[dict[str, float | str]]:
        """Compute mean |SHAP| global importance across a representative telemetry sample."""
        enriched = df if all(c in df.columns for c in ALL_MODEL_FEATURES) else add_domain_and_rolling_features(df)
        sample = enriched.sample(n=min(sample_size, len(enriched)), random_state=seed)
        shap_df = self.compute_shap_dataframe(sample)

        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        total_impact = float(mean_abs.sum()) or 1.0

        ranking = []
        for feature, val in mean_abs.items():
            ranking.append(
                {
                    "feature": str(feature),
                    "mean_abs_shap": round(float(val), 4),
                    "relative_importance_pct": round(float(val / total_impact * 100), 2),
                }
            )
        return ranking

    def explain_vehicles(
        self, df_rows: pd.DataFrame, top_k: int = 4
    ) -> list[dict]:
        """
        Produce structured local root-cause explanations for one or more vehicle rows.
        Returns top_k risk-increasing drivers per row for UI and AI Copilot consumption.
        """
        enriched = (
            df_rows.copy()
            if all(c in df_rows.columns for c in ALL_MODEL_FEATURES)
            else add_domain_and_rolling_features(df_rows)
        )
        probs = self.pipeline.predict_proba(enriched[ALL_MODEL_FEATURES])[:, 1]
        shap_df = self.compute_shap_dataframe(enriched)

        explanations: list[dict] = []
        for idx in range(len(enriched)):
            row = enriched.iloc[idx]
            row_shap = shap_df.iloc[idx].sort_values(ascending=False)

            top_drivers = []
            for feat_name, shap_val in row_shap.head(top_k).items():
                raw_val = row[feat_name]
                if isinstance(raw_val, (float, np.floating)):
                    formatted_val = round(float(raw_val), 2)
                elif isinstance(raw_val, (int, np.integer)):
                    formatted_val = int(raw_val)
                else:
                    formatted_val = str(raw_val)

                direction = "increases risk" if shap_val > 0 else "reduces risk"
                top_drivers.append(
                    {
                        "feature": str(feat_name),
                        "value": formatted_val,
                        "shap_impact": round(float(shap_val), 4),
                        "summary": f"{feat_name}={formatted_val} ({direction}, SHAP {shap_val:+.3f})",
                    }
                )

            explanations.append(
                {
                    "vehicle_id": str(row.get("Vehicle_ID", f"ROW-{idx}")),
                    "timestamp": str(row.get("Timestamp", "")),
                    "vehicle_type": str(row.get("Vehicle_Type", "")),
                    "route_corridor": str(row.get("Route_Corridor", "")),
                    "failure_prob_72h": round(float(probs[idx]), 4),
                    "top_risk_drivers": top_drivers,
                }
            )
        return explanations


def generate_and_save_global_shap(sample_size: int = 2500) -> Path:
    """Compute global feature importance on held-out test telemetry and persist JSON."""
    raw_df = pd.read_parquet(PATHS.raw_data / "synthetic_fleet_history.parquet")
    enriched = add_domain_and_rolling_features(raw_df)
    # Use the final 15% chronological test window so importance reflects unseen operations
    test_slice = enriched.iloc[int(len(enriched) * 0.85) :].reset_index(drop=True)

    explainer = FleetShapExplainer.from_saved_champion()
    ranking = explainer.compute_global_importance(test_slice, sample_size=sample_size)

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.models_dir / "global_shap_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sample_size": min(sample_size, len(test_slice)), "global_feature_importance": ranking}, f, indent=2)
    return out_path


if __name__ == "__main__":
    saved = generate_and_save_global_shap()
    print(f"Saved global SHAP importance to: {saved}")
    with open(saved, encoding="utf-8") as f:
        data = json.load(f)
    print("Top 10 Global Risk Drivers:")
    for item in data["global_feature_importance"][:10]:
        print(f"  {item['feature']:<30} | mean|SHAP|={item['mean_abs_shap']:.4f} ({item['relative_importance_pct']}%)")