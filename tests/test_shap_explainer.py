import joblib
from fleetguard.data.synthetic_generator import generate_historical_telemetry
from fleetguard.explainability.shap_explainer import FleetShapExplainer
from fleetguard.features.engineering import (
    ALL_MODEL_CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES,
    ALL_MODEL_NUMERICAL_FEATURES,
    HORIZON_TARGET_COL,
    add_domain_and_rolling_features,
    add_forward_failure_horizon,
)
from fleetguard.features.preprocessing import build_preprocessor_for_features
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def test_shap_explainer_global_and_local_attributions(tmp_path):
    df = generate_historical_telemetry(num_vehicles=15, steps_per_vehicle=25, seed=77)
    enriched = add_forward_failure_horizon(add_domain_and_rolling_features(df), horizon_steps=5).dropna(
        subset=[HORIZON_TARGET_COL]
    )

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor_for_features(
                    ALL_MODEL_NUMERICAL_FEATURES, ALL_MODEL_CATEGORICAL_FEATURES, scale_numeric=False
                ),
            ),
            ("classifier", XGBClassifier(n_estimators=20, max_depth=3, random_state=42)),
        ]
    )
    pipe.fit(enriched[ALL_MODEL_FEATURES], enriched[HORIZON_TARGET_COL].astype(int))

    temp_model = tmp_path / "temp_champion.joblib"
    joblib.dump(pipe, temp_model)

    explainer = FleetShapExplainer.from_saved_champion(model_path=temp_model)
    global_imp = explainer.compute_global_importance(enriched, sample_size=80)
    assert len(global_imp) == len(ALL_MODEL_FEATURES)
    assert abs(sum(item["relative_importance_pct"] for item in global_imp) - 100.0) < 0.5

    local_explanations = explainer.explain_vehicles(enriched.iloc[:3], top_k=3)
    assert len(local_explanations) == 3
    assert len(local_explanations[0]["top_risk_drivers"]) == 3
    assert 0.0 <= local_explanations[0]["failure_prob_72h"] <= 1.0