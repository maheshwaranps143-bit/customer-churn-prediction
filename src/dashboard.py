"""
dashboard.py
------------
A simple Streamlit dashboard with three tabs:
  1. Predict Churn  - fill in one customer's details, get a prediction + SHAP explanation
  2. Model Performance - compare accuracy / precision / recall / F1 / ROC-AUC across models
  3. Global Feature Importance - which features matter most to the model overall

Run from the project root:
    streamlit run src/dashboard.py

Note: this dashboard calls the model directly (via explainability.py), so it
works even without the FastAPI server running. If you'd rather have the
dashboard talk to the REST API instead, swap the direct calls below for
`requests.post("http://127.0.0.1:8000/predict", json=payload)`.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainability import explain_instance_shap, global_feature_importance_shap, load_artifacts
from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = "models"

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction Dashboard")
st.caption("Predict which customers are likely to churn, and understand *why*.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def artifacts_available():
    return os.path.exists(f"{MODELS_DIR}/best_model.pkl")


if not artifacts_available():
    st.error(
        "No trained model found. Run `python src/train_model.py` from the project "
        "root before launching this dashboard."
    )
    st.stop()

pipeline, feature_names, background_sample = get_artifacts()

tab_predict, tab_performance, tab_importance = st.tabs(
    ["🔮 Predict Churn", "📊 Model Performance", "🧠 Global Feature Importance"]
)

# ---------------------------------------------------------------- Tab 1 ----
with tab_predict:
    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["Yes", "No"])
        tenure_months = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])

    with col3:
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )
        monthly_charges = st.number_input("Monthly charges ($)", 0.0, 300.0, 70.0)
        total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, 840.0)
        num_support_calls = st.slider("Support calls (last period)", 0, 15, 1)

    if st.button("Predict churn", type="primary"):
        customer = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
        }
        X = pd.DataFrame([customer])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        proba = float(pipeline.predict_proba(X)[0, 1])
        prediction = "Yes" if proba >= 0.5 else "No"

        result_col, chart_col = st.columns([1, 2])
        with result_col:
            if prediction == "Yes":
                st.error(f"⚠️ Likely to churn ({proba:.1%} probability)")
            else:
                st.success(f"✅ Likely to stay ({proba:.1%} churn probability)")

        with chart_col:
            with st.spinner("Computing SHAP explanation..."):
                contributions = explain_instance_shap(pipeline, feature_names, background_sample, X)
            top_items = list(contributions.items())[:8]
            imp_df = pd.DataFrame(top_items, columns=["feature", "shap_value"])
            imp_df = imp_df.sort_values("shap_value")

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in imp_df["shap_value"]]
            ax.barh(imp_df["feature"], imp_df["shap_value"], color=colors)
            ax.set_xlabel("SHAP value (push toward churn →)")
            ax.set_title("Why this prediction?")
            st.pyplot(fig)
            st.caption("🔴 Red bars push toward churn · 🟢 Green bars push toward staying")

# ---------------------------------------------------------------- Tab 2 ----
with tab_performance:
    st.subheader("Model comparison")
    metrics_path = f"{MODELS_DIR}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_model = metrics.pop("best_model", None)
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))
        if best_model:
            st.info(f"🏆 Best model selected: **{best_model}** (highest ROC-AUC)")

        reports_dir = "reports"
        roc_path = f"{reports_dir}/roc_curves.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC curves - model comparison")
    else:
        st.warning("metrics.json not found. Run `python src/train_model.py` first.")

# ---------------------------------------------------------------- Tab 3 ----
with tab_importance:
    st.subheader("Which features matter most, across all customers?")
    sample_size = st.slider("Background sample size for SHAP", 20, 200, 100)
    with st.spinner("Computing global feature importance..."):
        importance_df = global_feature_importance_shap(
            pipeline, feature_names, background_sample, sample_size=sample_size
        )
    top_n = importance_df.head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_n["feature"][::-1], top_n["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    st.pyplot(fig)

    st.dataframe(importance_df, use_container_width=True)
