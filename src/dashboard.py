"""
dashboard.py
------------
Streamlit dashboard with four tabs:
  1. Individual prediction - enter one customer's details, get a churn
     probability and the reasons behind it
  2. Collective prediction - upload a CSV of many customers, get back the
     list of customers likely to churn, ranked by risk, as a download
  3. Model performance     - compare the trained models
  4. Feature importance    - which inputs matter most overall

Run from the project root:
    streamlit run src/dashboard.py

The dashboard calls the model directly (via explainability.py), so it
works without the FastAPI server running. All preprocessing happens inside
the saved pipeline: the app only ever passes raw customer values.
"""

import hashlib
import io
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from explainability import ChurnExplainer, load_artifacts
from preprocessing import (
    ALLOWED_VALUES,
    FEATURE_LABELS,
    ID_COL,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    OPTIONAL_ID_COLUMNS,
    template_dataframe,
    validate_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MAX_ROWS = 200_000
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40
RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}

st.set_page_config(
    page_title="Customer churn prediction",
    page_icon=":material/trending_down:",
    layout="wide",
)


@st.cache_resource
def get_model():
    pipeline, _, background = load_artifacts()
    return pipeline, background, ChurnExplainer(pipeline, background)


def risk_levels(probabilities) -> np.ndarray:
    return np.select(
        [probabilities >= HIGH_RISK, probabilities >= MEDIUM_RISK],
        ["High", "Medium"],
        default="Low",
    )


def read_uploaded_csv(data: bytes) -> pd.DataFrame:
    # Read every column as text so identifiers such as mobile numbers keep
    # their leading zeros and "+" signs. Numeric features are converted
    # during validation.
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The file's text encoding could not be read. Save it as UTF-8 CSV.")


def score_batch(data: bytes) -> dict:
    """Validate and score an uploaded file. Returns a dict for session state."""
    pipeline, _, explainer = get_model()
    try:
        raw = read_uploaded_csv(data)
    except pd.errors.EmptyDataError:
        return {"error": "The file is empty."}
    except (pd.errors.ParserError, ValueError) as exc:
        return {"error": f"The file could not be read as CSV: {exc}"}

    if raw.empty:
        return {"error": "The file has column headers but no customer rows."}
    if len(raw) > MAX_ROWS:
        return {"error": f"The file has {len(raw):,} rows. The limit is {MAX_ROWS:,} per upload; split it into smaller files."}

    valid, rejected, missing = validate_batch(raw)
    if missing:
        return {
            "error": "The file is missing these required columns: "
            + ", ".join(f"`{c}`" for c in missing)
            + ". Download the template to see the expected format."
        }

    scored = valid
    if not valid.empty:
        probabilities = pipeline.predict_proba(valid[MODEL_FEATURES])[:, 1]
        scored = valid.assign(
            churn_probability=probabilities.round(4),
            risk_level=risk_levels(probabilities),
            top_reasons=explainer.top_reasons(valid),
        ).sort_values("churn_probability", ascending=False)

    return {
        "error": None,
        "scored": scored,
        "rejected": rejected,
        "duplicate_ids": int(valid[ID_COL].duplicated().sum()),
    }


def contribution_chart(contributions: pd.Series) -> alt.Chart:
    df = pd.DataFrame(
        {
            "feature": [FEATURE_LABELS[f] for f in contributions.index],
            "contribution": contributions.to_numpy(),
        }
    )
    df["effect"] = np.where(df["contribution"] > 0, "Pushes towards churn", "Pushes towards staying")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Effect on churn risk"),
            y=alt.Y("feature:N", sort=list(df["feature"]), title=None),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(
                    domain=["Pushes towards churn", "Pushes towards staying"],
                    range=[RISK_COLORS["High"], RISK_COLORS["Low"]],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
        )
    )


st.title("Customer churn prediction")
st.caption("Find the customers likely to leave, and see why.")

if not (MODELS_DIR / "best_model.pkl").exists():
    st.error(
        "No trained model found. Run `python src/train_model.py` from the "
        "project root before launching this dashboard."
    )
    st.stop()

pipeline, background_sample, explainer = get_model()

tab_single, tab_batch, tab_performance, tab_importance = st.tabs(
    [
        ":material/person: Individual prediction",
        ":material/groups: Collective prediction",
        ":material/analytics: Model performance",
        ":material/insights: Feature importance",
    ]
)

# ------------------------------------------------ Individual prediction ----
with tab_single:
    with st.form("single_customer"):
        st.subheader("Customer details")
        left, right = st.columns(2)
        with left:
            contract = st.selectbox(FEATURE_LABELS["contract"], ALLOWED_VALUES["contract"])
            tenure_months = st.number_input(FEATURE_LABELS["tenure_months"], min_value=0, max_value=600, value=12, step=1)
            monthly_charges = st.number_input(FEATURE_LABELS["monthly_charges"], min_value=0.0, max_value=100000.0, value=70.0, step=5.0)
            num_support_calls = st.number_input(
                FEATURE_LABELS["num_support_calls"],
                min_value=0,
                max_value=100,
                value=1,
                step=1,
                help="Number of support calls the customer made recently.",
            )
        with right:
            internet_service = st.selectbox(FEATURE_LABELS["internet_service"], ALLOWED_VALUES["internet_service"], index=1)
            payment_method = st.selectbox(FEATURE_LABELS["payment_method"], ALLOWED_VALUES["payment_method"])
            tech_support = st.selectbox(FEATURE_LABELS["tech_support"], ALLOWED_VALUES["tech_support"], index=1)
            online_security = st.selectbox(FEATURE_LABELS["online_security"], ALLOWED_VALUES["online_security"], index=1)
        submitted = st.form_submit_button("Predict churn", type="primary", icon=":material/query_stats:")

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "tenure_months": tenure_months,
                    "monthly_charges": monthly_charges,
                    "num_support_calls": num_support_calls,
                    "contract": contract,
                    "internet_service": internet_service,
                    "payment_method": payment_method,
                    "tech_support": tech_support,
                    "online_security": online_security,
                }
            ]
        )
        probability = float(pipeline.predict_proba(customer[MODEL_FEATURES])[0, 1])
        level = str(risk_levels(np.array([probability]))[0])

        result_col, chart_col = st.columns([1, 2])
        with result_col:
            st.metric("Churn probability", f"{probability:.1%}")
            badge_color = {"High": "red", "Medium": "orange", "Low": "green"}[level]
            st.markdown(f"Risk level: :{badge_color}-badge[{level}]")
            if probability >= 0.5:
                st.error("Likely to churn", icon=":material/warning:")
            else:
                st.success("Likely to stay", icon=":material/check_circle:")
        with chart_col:
            st.markdown("**Why this prediction?**")
            st.altair_chart(contribution_chart(explainer.explain_one(customer)))

# ------------------------------------------------ Collective prediction ----
with tab_batch:
    st.subheader("Score a whole customer list")
    st.markdown(
        "Upload a CSV with one row per customer. The app checks every row, "
        "predicts each customer's churn probability, and returns the customers "
        "most likely to leave, riskiest first."
    )

    with st.expander("Required file format", icon=":material/description:"):
        format_rows = [
            {"Column": ID_COL, "Required": "Yes", "Accepted values": "Any unique identifier"},
        ]
        for col in OPTIONAL_ID_COLUMNS:
            format_rows.append({"Column": col, "Required": "No", "Accepted values": "Any text, shown in results only"})
        for col in NUMERIC_FEATURES:
            format_rows.append({"Column": col, "Required": "Yes", "Accepted values": "Number, 0 or more"})
        for col, values in ALLOWED_VALUES.items():
            format_rows.append({"Column": col, "Required": "Yes", "Accepted values": ", ".join(values)})
        st.dataframe(pd.DataFrame(format_rows), hide_index=True)
        st.caption(
            "Column names and category values are not case-sensitive. Any extra "
            "columns are kept in the results. Rows with missing or invalid values "
            "are listed separately instead of being guessed."
        )
        st.download_button(
            "Download template CSV",
            data=template_dataframe().to_csv(index=False),
            file_name="churn_upload_template.csv",
            mime="text/csv",
            on_click="ignore",
            icon=":material/download:",
        )

    uploaded = st.file_uploader("Customer file (CSV)", type="csv")
    st.caption(
        ":material/lock: Uploaded data is processed in memory for your session "
        "only and is never written to disk."
    )

    if uploaded is None:
        st.info("Upload a CSV file to begin.", icon=":material/upload_file:")
    else:
        data = uploaded.getvalue()
        file_key = hashlib.sha256(data).hexdigest()
        if st.session_state.get("batch_key") != file_key:
            with st.spinner("Checking and scoring customers..."):
                st.session_state.batch_result = score_batch(data)
            st.session_state.batch_key = file_key
        result = st.session_state.batch_result

        if result["error"]:
            st.error(result["error"], icon=":material/error:")
        else:
            scored = result["scored"]
            rejected = result["rejected"]

            if result["duplicate_ids"]:
                st.warning(
                    f"{result['duplicate_ids']} rows reuse a customer_id already seen in the file. "
                    "They were scored, but check the source data.",
                    icon=":material/content_copy:",
                )

            if scored.empty:
                st.error("No rows could be scored. See the problems listed below.", icon=":material/error:")
            else:
                threshold_pct = st.slider(
                    "Flag customers as likely to churn at or above this probability",
                    min_value=5,
                    max_value=95,
                    value=50,
                    step=5,
                    format="%d%%",
                )
                flagged = scored["churn_probability"] >= threshold_pct / 100
                results = scored.assign(likely_to_churn=np.where(flagged, "Yes", "No"))

                with st.container(horizontal=True):
                    st.metric("Customers scored", f"{len(results):,}", border=True)
                    st.metric(
                        "Likely to churn",
                        f"{int(flagged.sum()):,}",
                        delta=f"{flagged.mean():.1%} of customers",
                        delta_color="off",
                        border=True,
                    )
                    st.metric("High risk (70%+)", f"{int((results['risk_level'] == 'High').sum()):,}", border=True)
                    st.metric("Rows rejected", f"{len(rejected):,}", border=True)

                risk_counts = (
                    results["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).rename_axis("risk_level").reset_index(name="customers")
                )
                st.altair_chart(
                    alt.Chart(risk_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("customers:Q", title="Customers"),
                        y=alt.Y("risk_level:N", sort=["High", "Medium", "Low"], title="Risk level"),
                        color=alt.Color(
                            "risk_level:N",
                            scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values())),
                            legend=None,
                        ),
                        tooltip=["risk_level", "customers"],
                    )
                    .properties(height=140)
                )

                view = st.segmented_control(
                    "Show",
                    ["Likely to churn", "All customers"],
                    default="Likely to churn",
                    required=True,
                )
                shown = results[flagged] if view == "Likely to churn" else results

                id_cols = [ID_COL] + [c for c in OPTIONAL_ID_COLUMNS if c in results.columns]
                extra_cols = [
                    c
                    for c in results.columns
                    if c not in id_cols + MODEL_FEATURES + ["churn_probability", "risk_level", "top_reasons", "likely_to_churn"]
                ]
                ordered = id_cols + ["churn_probability", "risk_level", "likely_to_churn", "top_reasons"] + MODEL_FEATURES + extra_cols

                st.dataframe(
                    shown[ordered],
                    hide_index=True,
                    column_config={
                        ID_COL: st.column_config.TextColumn("Customer ID", pinned=True),
                        "churn_probability": st.column_config.ProgressColumn(
                            "Churn probability", format="percent", min_value=0, max_value=1
                        ),
                        "risk_level": "Risk",
                        "likely_to_churn": "Likely to churn",
                        "top_reasons": st.column_config.TextColumn("Main reasons", width="large"),
                        **{f: FEATURE_LABELS[f] for f in MODEL_FEATURES},
                    },
                )
                if shown.empty:
                    st.info("No customers are at or above the selected probability.")

                with st.container(horizontal=True):
                    st.download_button(
                        f"Download likely churners ({int(flagged.sum()):,})",
                        data=results[flagged][ordered].to_csv(index=False),
                        file_name="likely_churners.csv",
                        mime="text/csv",
                        on_click="ignore",
                        type="primary",
                        icon=":material/download:",
                    )
                    st.download_button(
                        f"Download all results ({len(results):,})",
                        data=results[ordered].to_csv(index=False),
                        file_name="churn_predictions_all.csv",
                        mime="text/csv",
                        on_click="ignore",
                        icon=":material/download:",
                    )

            if not rejected.empty:
                with st.expander(f"{len(rejected):,} rows could not be scored", icon=":material/report:"):
                    st.dataframe(rejected, hide_index=True)
                    st.download_button(
                        "Download rejected rows",
                        data=rejected.to_csv(index=False),
                        file_name="rejected_rows.csv",
                        mime="text/csv",
                        on_click="ignore",
                        icon=":material/download:",
                    )

# ---------------------------------------------------- Model performance ----
with tab_performance:
    st.subheader("Model comparison")
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        best_model = metrics.pop("best_model", None)
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"))
        if best_model:
            st.info(
                f"Best model: **{best_model}**, selected by highest ROC-AUC. "
                f"Trained on {len(MODEL_FEATURES)} input features.",
                icon=":material/emoji_events:",
            )
        roc_path = REPORTS_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC curves - model comparison")
    else:
        st.warning("metrics.json not found. Run `python src/train_model.py` first.")

# --------------------------------------------------- Feature importance ----
with tab_importance:
    st.subheader("Which inputs matter most, across all customers?")
    importance = explainer.global_importance(background_sample)
    st.altair_chart(
        alt.Chart(importance)
        .mark_bar(color="steelblue")
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Average effect on churn risk"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
        )
    )
    st.caption(
        "Average size of each input's effect on the prediction (mean absolute "
        "SHAP value), measured over a sample of training customers."
    )
