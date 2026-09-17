"""
api.py
------
FastAPI REST API exposing the trained churn model.

Endpoints:
    GET  /health              -> simple health check
    POST /predict             -> churn probability + reasons for one customer
    GET  /feature-importance  -> global feature importance for the model
    GET  /metrics             -> saved evaluation metrics for all trained models

Run from the project root:
    uvicorn src.api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow running this file both as `python src/api.py` and as
# `uvicorn src.api:app` by making sure src/ is on the path.
sys.path.append(str(Path(__file__).resolve().parent))

from explainability import MODELS_DIR, ChurnExplainer, load_artifacts
from preprocessing import FEATURE_LABELS, MODEL_FEATURES

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts once at startup (not on every request).
    try:
        pipeline, _, background_sample = load_artifacts()
        state["pipeline"] = pipeline
        state["background"] = background_sample
        state["explainer"] = ChurnExplainer(pipeline, background_sample)
        print("Model artifacts loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model artifacts not found. Run `python src/train_model.py` first.")
    yield
    state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn and explains predictions with SHAP.",
    version="2.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    """One customer's raw feature values. Invalid values are rejected with HTTP 422."""

    tenure_months: int = Field(..., ge=0, examples=[12])
    monthly_charges: float = Field(..., ge=0, examples=[85.5])
    num_support_calls: int = Field(..., ge=0, examples=[3])
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., examples=["Month-to-month"])
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(..., examples=["Fiber optic"])
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"] = Field(
        ..., examples=["Electronic check"]
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(..., examples=["No"])
    online_security: Literal["Yes", "No", "No internet service"] = Field(..., examples=["No"])


def _require_model():
    if "pipeline" not in state:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train_model.py` to train and save a model first.",
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    """Predicts churn probability for one customer and explains the prediction with SHAP."""
    _require_model()

    X = pd.DataFrame([customer.model_dump()])[MODEL_FEATURES]
    churn_probability = float(state["pipeline"].predict_proba(X)[0, 1])
    contributions = state["explainer"].explain_one(X)

    return {
        "churn_prediction": "Yes" if churn_probability >= 0.5 else "No",
        "churn_probability": round(churn_probability, 4),
        "top_reasons": [
            {"feature": FEATURE_LABELS[feature], "impact": round(float(value), 4)}
            for feature, value in list(contributions.items())[:5]
        ],
    }


@app.get("/feature-importance")
def feature_importance():
    """Global feature importance (mean absolute SHAP value) over a background sample."""
    _require_model()
    return state["explainer"].global_importance(state["background"]).to_dict(orient="records")


@app.get("/metrics")
def get_metrics():
    """Saved evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) for all models."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Run train_model.py first.")
    return json.loads(metrics_path.read_text())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
