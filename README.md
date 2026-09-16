# Customer Churn Prediction System with Explainable AI

An intermediate-level, end-to-end machine learning project that predicts whether
a customer is likely to leave (churn) a subscription service — and **explains why**
each prediction was made, using SHAP and LIME.

---

## ✨ Features

- **Data preprocessing**: missing-value imputation, categorical encoding, scaling
- **Feature engineering** via a reusable scikit-learn `ColumnTransformer` pipeline
- **Multiple models**: Logistic Regression, Random Forest, Gradient Boosting — automatically compared and the best one (by ROC-AUC) is selected
- **Imbalanced-data handling** via `class_weight="balanced"`
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Explainability layer**: SHAP (global + local explanations) and LIME (local, cross-check)
- **REST API** (FastAPI): `/predict`, `/feature-importance`, `/metrics`, `/health`
- **Streamlit dashboard**: interactive churn prediction + visual explanations
- **Docker support** for one-command deployment

---

## 🗂 Project Structure

```
churn_project/
├── data/
│   ├── generate_sample_data.py   # creates the synthetic sample dataset
│   └── customer_churn.csv        # sample dataset (3000 customers)
├── notebooks/
│   ├── 01_data_exploration.ipynb # EDA: churn rate, correlations, boxplots
│   └── 02_model_training.ipynb   # interactive walkthrough of training
├── src/
│   ├── preprocessing.py          # cleaning, encoding, scaling, train/test split
│   ├── train_model.py            # trains + evaluates + saves the best model
│   ├── explainability.py         # SHAP + LIME explanation logic
│   ├── api.py                    # FastAPI REST API
│   └── dashboard.py              # Streamlit dashboard
├── models/                       # created after training (pkl artifacts)
├── reports/                      # created after training (evaluation plots)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quickstart (local, no Docker)

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the sample dataset (skip if you already have data/customer_churn.csv)
python data/generate_sample_data.py

# 4. Train the models (this saves models/best_model.pkl + evaluation reports)
python src/train_model.py

# 5a. Launch the REST API
uvicorn src.api:app --reload --port 8000
# then open http://127.0.0.1:8000/docs for interactive Swagger UI

# 5b. OR launch the Streamlit dashboard (in a separate terminal)
streamlit run src/dashboard.py
```

> **Note:** Run all commands from the project **root** folder (`churn_project/`),
> since the scripts use relative paths like `data/...` and `models/...`.

---

## 🐳 Quickstart (Docker)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The image trains the model at build time, so the API is ready to serve immediately.
To run the dashboard instead of the API:

```bash
docker run -p 8501:8501 churn-api streamlit run src/dashboard.py --server.address 0.0.0.0
```

---

## 🔌 API Reference

| Method | Endpoint               | Description                                              |
|--------|-------------------------|------------------------------------------------------------|
| GET    | `/health`               | Health check                                              |
| POST   | `/predict`               | Predict churn probability + top SHAP reasons for one customer |
| GET    | `/feature-importance`   | Global feature importance (mean absolute SHAP value)       |
| GET    | `/metrics`               | Evaluation metrics for every trained model                 |

Example `/predict` request body:

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure_months": 3,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 95.5,
  "total_charges": 286.5,
  "num_support_calls": 4
}
```

Example response:

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.9786,
  "top_reasons": [
    {"feature": "tenure_months", "impact": 0.1096},
    {"feature": "num_support_calls", "impact": 0.0869},
    {"feature": "contract_Month-to-month", "impact": 0.0809},
    {"feature": "monthly_charges", "impact": 0.056},
    {"feature": "internet_service_Fiber optic", "impact": 0.0372}
  ]
}
```

---

## 🗄 Using a real SQL database instead of the sample CSV

`src/preprocessing.py` includes a `load_from_sql()` helper built on SQLAlchemy.
Point it at your customer database instead of the CSV:

```python
from preprocessing import load_from_sql

query = """
    SELECT customer_id, gender, senior_citizen, partner, dependents,
           tenure_months, phone_service, multiple_lines, internet_service,
           online_security, tech_support, streaming_tv, contract,
           paperless_billing, payment_method, monthly_charges,
           total_charges, num_support_calls, churn
    FROM customers
"""
df = load_from_sql("postgresql://user:password@host:5432/mydb", query)
```

Then pass `df` into `clean_data()` / `prepare_train_test_split()` as usual.

---

## 🧠 How the explainability layer works

- **SHAP** (SHapley Additive exPlanations) assigns every feature a contribution
  value showing how much it pushed a specific prediction toward "churn" or
  "no churn," based on cooperative game theory. Used for both:
  - **Local explanations** — "why did *this* customer get flagged?" (`/predict`)
  - **Global explanations** — "what matters most across *all* customers?" (`/feature-importance`)
- **LIME** (Local Interpretable Model-agnostic Explanations) is included as a
  second, independent method for explaining individual predictions — useful as
  a sanity check against SHAP, since the two methods use different underlying
  approaches but should broadly agree.

---

## 📈 Retraining on your own data

1. Replace `data/customer_churn.csv` with your own dataset, keeping the same
   column names (or update `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in
   `src/preprocessing.py` to match your columns).
2. Run `python src/train_model.py` again.
3. Restart the API / dashboard — they always load the latest `models/best_model.pkl`.

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/customer_churn.csv` | Run scripts from the project root, not from inside `src/` |
| API says "Model not loaded" | Run `python src/train_model.py` first to create `models/best_model.pkl` |
| SHAP explanations are slow | Reduce `sample_size` in `/feature-importance` or the background sample size in `train_model.py` |
