"""
api.py
------
FastAPI REST API exposing the trained churn model.

Endpoints:
    GET  /health              -> simple health check
    POST /predict              -> predict churn probability + SHAP explanation for one customer
    GET  /feature-importance   -> global SHAP feature importance for the whole model
    GET  /metrics               -> saved evaluation metrics for all trained models

Run from the project root:
    uvicorn src.api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.
"""

import json
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow running this file both as `python src/api.py` and as
# `uvicorn src.api:app` by making sure src/ is on the path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainability import explain_instance_shap, global_feature_importance_shap, load_artifacts
from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = "models"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn and explains predictions with SHAP.",
    version="1.0.0",
)

# Load model artifacts once at startup (not on every request - much faster).
pipeline, feature_names, background_sample = None, None, None


@app.on_event("startup")
def load_model_artifacts():
    global pipeline, feature_names, background_sample
    try:
        pipeline, feature_names, background_sample = load_artifacts()
        print("Model artifacts loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model artifacts not found. Run `python src/train_model.py` first.")


class CustomerFeatures(BaseModel):
    """Schema for a single customer's raw (un-encoded) feature values."""

    gender: str = Field(..., example="Female")
    senior_citizen: int = Field(..., example=0)
    partner: str = Field(..., example="Yes")
    dependents: str = Field(..., example="No")
    tenure_months: int = Field(..., example=12)
    phone_service: str = Field(..., example="Yes")
    multiple_lines: str = Field(..., example="No")
    internet_service: str = Field(..., example="Fiber optic")
    online_security: str = Field(..., example="No")
    tech_support: str = Field(..., example="No")
    streaming_tv: str = Field(..., example="Yes")
    contract: str = Field(..., example="Month-to-month")
    paperless_billing: str = Field(..., example="Yes")
    payment_method: str = Field(..., example="Electronic check")
    monthly_charges: float = Field(..., example=85.5)
    total_charges: float = Field(..., example=1026.0)
    num_support_calls: int = Field(..., example=3)


def _ensure_model_loaded():
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train_model.py` to train and save a model first.",
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    """Predicts churn probability for one customer and explains the prediction with SHAP."""
    _ensure_model_loaded()

    X = pd.DataFrame([customer.dict()])
    X = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]  # enforce correct column order

    churn_probability = float(pipeline.predict_proba(X)[0, 1])
    churn_prediction = "Yes" if churn_probability >= 0.5 else "No"

    shap_contributions = explain_instance_shap(pipeline, feature_names, background_sample, X)
    top_reasons = [
        {"feature": feat, "impact": round(float(val), 4)}
        for feat, val in list(shap_contributions.items())[:5]
    ]

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(churn_probability, 4),
        "top_reasons": top_reasons,
    }


@app.get("/feature-importance")
def feature_importance(sample_size: int = 100):
    """Returns global feature importance (mean absolute SHAP value) across a background sample."""
    _ensure_model_loaded()

    importance_df = global_feature_importance_shap(
        pipeline, feature_names, background_sample, sample_size=sample_size
    )
    return importance_df.to_dict(orient="records")


@app.get("/metrics")
def get_metrics():
    """Returns the saved evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) for all models."""
    metrics_path = f"{MODELS_DIR}/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="metrics.json not found. Run train_model.py first.")
    with open(metrics_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
