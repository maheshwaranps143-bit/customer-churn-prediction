"""
explainability.py
------------------
Wraps SHAP and LIME so the API and dashboard can answer:
  "WHY did the model predict this customer will churn?"

Two explanation styles are provided:
  - SHAP: fast, consistent, great for both global (whole model) and
    local (single customer) explanations. Used as the primary method.
  - LIME: an alternative local explanation method, useful for
    cross-checking SHAP on a single prediction.

Run directly (`python src/explainability.py`) to sanity-check that a
trained model + saved background sample produce a valid explanation.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = "models"


def load_artifacts():
    pipeline = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    background_sample = joblib.load(f"{MODELS_DIR}/background_sample.pkl")
    return pipeline, feature_names, background_sample


def _transform(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Applies just the preprocessing step of the pipeline (not the model)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    # OneHotEncoder output can be a sparse matrix; SHAP/LIME want dense arrays.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed


def build_shap_explainer(pipeline, background_sample: pd.DataFrame):
    """
    Builds a SHAP explainer around the *model only* (post-preprocessing),
    using a small background sample for the expected-value baseline.
    Uses the generic Explainer, which auto-picks a fast algorithm
    (TreeExplainer for tree models, LinearExplainer for linear models, etc).
    """
    model = pipeline.named_steps["model"]
    background_transformed = _transform(pipeline, background_sample)
    explainer = shap.Explainer(model.predict_proba, background_transformed)
    return explainer


def explain_instance_shap(pipeline, feature_names, background_sample, X_instance: pd.DataFrame):
    """
    Returns a dict of {feature_name: shap_value} for a single customer,
    sorted by absolute impact (most influential feature first).
    """
    explainer = build_shap_explainer(pipeline, background_sample)
    X_transformed = _transform(pipeline, X_instance)

    shap_values = explainer(X_transformed)
    # shap_values.values shape: (n_samples, n_features, n_classes) for predict_proba
    # We want the contribution toward the "Yes churn" class (index 1).
    values = shap_values.values[0]
    if values.ndim == 2:
        values = values[:, 1]

    contributions = dict(zip(feature_names, values))
    sorted_contributions = dict(
        sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )
    return sorted_contributions


def explain_instance_lime(pipeline, feature_names, background_sample, X_instance: pd.DataFrame, num_features=10):
    """
    Returns a list of (feature_description, weight) tuples from LIME for
    a single customer. LIME works directly on the transformed numeric
    feature space, same as our SHAP setup.
    """
    background_transformed = _transform(pipeline, background_sample)
    instance_transformed = _transform(pipeline, X_instance)[0]

    explainer = LimeTabularExplainer(
        training_data=background_transformed,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        mode="classification",
    )

    model = pipeline.named_steps["model"]
    explanation = explainer.explain_instance(
        instance_transformed, model.predict_proba, num_features=num_features
    )
    return explanation.as_list()


def global_feature_importance_shap(pipeline, feature_names, background_sample, sample_size=100):
    """
    Computes mean absolute SHAP values across a sample of customers, giving
    a GLOBAL ranking of which features matter most to the model overall
    (used by the /feature-importance API endpoint and the dashboard).
    """
    explainer = build_shap_explainer(pipeline, background_sample)
    sample = background_sample.sample(min(sample_size, len(background_sample)), random_state=42)
    X_transformed = _transform(pipeline, sample)

    shap_values = explainer(X_transformed)
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_shap = np.abs(values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)
    return importance_df


if __name__ == "__main__":
    from preprocessing import load_data, prepare_train_test_split

    pipeline, feature_names, background_sample = load_artifacts()

    df = load_data()
    _, X_test, _, _ = prepare_train_test_split(df)
    sample_customer = X_test.iloc[[0]]

    print("Explaining one sample customer with SHAP:")
    shap_result = explain_instance_shap(pipeline, feature_names, background_sample, sample_customer)
    for feat, val in list(shap_result.items())[:8]:
        print(f"  {feat:35s} {val:+.4f}")

    print("\nGlobal feature importance (mean |SHAP value|):")
    global_imp = global_feature_importance_shap(pipeline, feature_names, background_sample)
    print(global_imp.head(8).to_string(index=False))
