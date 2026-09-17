"""
explainability.py
------------------
Answers "WHY did the model predict this customer will churn?"

ChurnExplainer wraps SHAP around the trained pipeline. It is built once
and reused, and it picks SHAP's fast exact algorithm for the model type
(LinearExplainer for linear models, TreeExplainer for tree ensembles), so
explaining thousands of customers takes seconds rather than hours.

Contributions are reported per *original* feature ("contract"), not per
one-hot column ("contract_Month-to-month"). SHAP values are additive, so
summing the one-hot columns of a feature gives that feature's total effect.

A LIME helper is kept as an independent cross-check on single predictions.

Run directly (`python src/explainability.py`) to sanity-check the saved
artifacts.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    get_feature_names,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    background_sample = joblib.load(MODELS_DIR / "background_sample.pkl")
    return pipeline, feature_names, background_sample


def _dense(matrix) -> np.ndarray:
    # OneHotEncoder output can be a sparse matrix; SHAP/LIME want dense arrays.
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class ChurnExplainer:
    """SHAP explanations for a fitted preprocessing + model pipeline."""

    def __init__(self, pipeline, background_sample: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]
        self.encoded_names = get_feature_names(self.preprocessor)
        self._source_index = self._map_encoded_to_source()

        background = self._transform(background_sample)
        if hasattr(self.model, "coef_"):
            masker = shap.maskers.Independent(background, max_samples=len(background))
            self._explainer = shap.LinearExplainer(self.model, masker)
        elif hasattr(self.model, "estimators_"):
            self._explainer = shap.TreeExplainer(self.model)
        else:
            # Model-agnostic fallback: correct for any model, but slow.
            self._explainer = shap.Explainer(self.model.predict_proba, background)

    def _map_encoded_to_source(self) -> np.ndarray:
        """For each encoded column, the index of the MODEL_FEATURES entry it came from."""
        onehot = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        source = [MODEL_FEATURES.index(f) for f in NUMERIC_FEATURES]
        for feature, categories in zip(CATEGORICAL_FEATURES, onehot.categories_):
            source += [MODEL_FEATURES.index(feature)] * len(categories)
        return np.array(source)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return _dense(self.preprocessor.transform(X[MODEL_FEATURES]))

    def encoded_contributions(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values per encoded column, shape (rows, encoded columns).
        Positive values push towards churn."""
        values = self._explainer(self._transform(X)).values
        if values.ndim == 3:
            # Per-class output: keep the contribution towards "churn".
            values = values[:, :, 1]
        return values

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values summed per original feature, shape (rows, 8)."""
        encoded = self.encoded_contributions(X)
        summed = np.zeros((encoded.shape[0], len(MODEL_FEATURES)))
        np.add.at(summed.T, self._source_index, encoded.T)
        return pd.DataFrame(summed, columns=MODEL_FEATURES, index=X.index)

    def explain_one(self, X_row: pd.DataFrame) -> pd.Series:
        """Contributions for a single customer, most influential first."""
        row = self.contributions(X_row).iloc[0]
        return row.reindex(row.abs().sort_values(ascending=False).index)

    def top_reasons(self, X: pd.DataFrame, k: int = 2) -> list:
        """
        For each customer, the k features pushing hardest towards churn,
        written as readable text, e.g. "Contract: Month-to-month".
        Customers with nothing pushing towards churn get "-".
        """
        contrib = self.contributions(X)
        values = contrib.to_numpy()
        order = np.argsort(-values, axis=1)[:, :k]
        raw = X[MODEL_FEATURES].to_numpy()

        reasons = []
        for row in range(len(values)):
            parts = []
            for col in order[row]:
                if values[row, col] <= 0:
                    break
                feature = MODEL_FEATURES[col]
                parts.append(f"{FEATURE_LABELS[feature]}: {_format_value(feature, raw[row, col])}")
            reasons.append("; ".join(parts) if parts else "-")
        return reasons

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute contribution per feature across X, largest first."""
        mean_abs = self.contributions(X).abs().mean()
        return (
            pd.DataFrame(
                {
                    "feature": [FEATURE_LABELS[f] for f in mean_abs.index],
                    "mean_abs_shap": mean_abs.to_numpy(),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def _format_value(feature: str, value) -> str:
    if feature == "monthly_charges":
        return f"{float(value):.2f}"
    if feature in NUMERIC_FEATURES:
        return f"{float(value):g}"
    return str(value)


def explain_instance_lime(pipeline, background_sample, X_instance: pd.DataFrame, num_features=10):
    """
    Returns a list of (feature_description, weight) tuples from LIME for
    a single customer. LIME works on the transformed numeric feature
    space, the same space SHAP uses.
    """
    from lime.lime_tabular import LimeTabularExplainer

    preprocessor = pipeline.named_steps["preprocessor"]
    explainer = LimeTabularExplainer(
        training_data=_dense(preprocessor.transform(background_sample[MODEL_FEATURES])),
        feature_names=get_feature_names(preprocessor),
        class_names=["No Churn", "Churn"],
        mode="classification",
    )
    instance = _dense(preprocessor.transform(X_instance[MODEL_FEATURES]))[0]
    explanation = explainer.explain_instance(
        instance, pipeline.named_steps["model"].predict_proba, num_features=num_features
    )
    return explanation.as_list()


if __name__ == "__main__":
    from preprocessing import load_data, prepare_train_test_split

    pipeline, _, background_sample = load_artifacts()
    explainer = ChurnExplainer(pipeline, background_sample)

    _, X_test, _, _ = prepare_train_test_split(load_data())
    sample_customer = X_test.iloc[[0]]

    print("Explaining one sample customer with SHAP:")
    for feature, value in explainer.explain_one(sample_customer).items():
        print(f"  {FEATURE_LABELS[feature]:20s} {value:+.4f}")

    print("\nTop reasons for the first five test customers:")
    for reason in explainer.top_reasons(X_test.head(5)):
        print("  ", reason)

    print("\nGlobal feature importance (mean |SHAP value|):")
    print(explainer.global_importance(X_test).to_string(index=False))
