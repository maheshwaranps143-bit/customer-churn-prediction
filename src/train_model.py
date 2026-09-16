"""
train_model.py
---------------
Trains several classifiers on the churn dataset, evaluates them,
handles class imbalance, picks the best model by ROC-AUC, and saves:

    models/best_model.pkl        -> fitted Pipeline (preprocessor + model)
    models/feature_names.pkl     -> feature names for SHAP / importance plots
    models/metrics.json          -> evaluation metrics for every model
    reports/*.png                -> confusion matrix, ROC curve, feature importance

Run from the project root:
    python src/train_model.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import (
    build_preprocessor,
    get_feature_names,
    load_data,
    prepare_train_test_split,
)

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def get_candidate_models() -> dict:
    """
    Returns the models we want to compare.

    class_weight="balanced" (Logistic Regression, Random Forest) and
    a manual scale on Gradient Boosting are how we handle the
    class imbalance (churners are usually the minority class) WITHOUT
    throwing away data via undersampling.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name, out_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curves(fitted_pipelines: dict, X_test, y_test, out_path):
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for name, pipeline in fitted_pipelines.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
    plt.title("ROC Curves - Model Comparison")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(pipeline: Pipeline, feature_names, model_name, out_path, top_n=15):
    """Plots built-in feature importance (tree models) or coefficients (logistic regression)."""
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print(f"No native feature importance available for {model_name}, skipping plot.")
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="steelblue")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load + split data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    # 2. Train + evaluate every candidate model
    all_metrics = {}
    fitted_pipelines = {}
    preprocessor_template = build_preprocessor()

    for name, model in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        all_metrics[name] = evaluate_model(name, pipeline, X_test, y_test)

        # Save a confusion matrix per model
        y_pred = pipeline.predict(X_test)
        plot_confusion_matrix(
            y_test, y_pred, name, f"{REPORTS_DIR}/confusion_matrix_{name}.png"
        )

    # 3. Pick the best model by ROC-AUC (a good single metric for imbalanced churn data)
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {all_metrics[best_name]['roc_auc']})")

    # 4. Save comparison plots
    plot_roc_curves(fitted_pipelines, X_test, y_test, f"{REPORTS_DIR}/roc_curves.png")

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    plot_feature_importance(
        best_pipeline,
        feature_names,
        best_name,
        f"{REPORTS_DIR}/feature_importance_{best_name}.png",
    )

    # 5. Persist everything the API / dashboard will need
    joblib.dump(best_pipeline, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.pkl")
    joblib.dump(best_name, f"{MODELS_DIR}/best_model_name.pkl")
    # Keep a small validation sample around; SHAP's explainer needs a
    # background dataset to compute explanations against.
    joblib.dump(X_train.sample(min(200, len(X_train)), random_state=42), f"{MODELS_DIR}/background_sample.pkl")

    all_metrics["best_model"] = best_name
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved best model + artifacts to '{MODELS_DIR}/'")
    print(f"Saved evaluation plots to '{REPORTS_DIR}/'")


if __name__ == "__main__":
    main()
