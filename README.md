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
- **Streamlit dashboard**: individual prediction, plus **collective prediction** — upload a customer CSV and download the list of customers likely to churn, ranked by risk, with the main reasons for each
- **Eight input features**, chosen because they are the ones that drive churn in the data
- **Docker support** for one-command deployment

---

## 🗂 Project Structure

```
churn_project/
├── data/
│   ├── generate_sample_data.py   # creates the synthetic sample dataset
│   ├── customer_churn.csv        # training dataset (3000 customers)
│   └── sample_customers.csv      # example file for collective prediction
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
  "tenure_months": 8,
  "monthly_charges": 85.5,
  "num_support_calls": 3,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check",
  "tech_support": "No",
  "online_security": "No"
}
```

Example response:

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.9639,
  "top_reasons": [
    {"feature": "Contract", "impact": 1.048},
    {"feature": "Tenure (months)", "impact": 0.9397},
    {"feature": "Internet service", "impact": 0.4878},
    {"feature": "Payment method", "impact": 0.4064},
    {"feature": "Support calls", "impact": 0.4027}
  ]
}
```

Values outside the accepted set (for example `"contract": "Weekly"` or a
negative tenure) are rejected with HTTP 422 instead of producing a guess.

---

## 📋 Input features

| Feature | Accepted values | Why it matters |
|---|---|---|
| `contract` | Month-to-month, One year, Two year | Strongest driver: monthly customers can leave at any time |
| `tenure_months` | whole number, 0 or more | Long-standing customers are less likely to leave |
| `monthly_charges` | number, 0 or more | Higher bills raise churn risk |
| `num_support_calls` | whole number, 0 or more | Repeated calls signal unresolved problems |
| `internet_service` | DSL, Fiber optic, No | Fibre customers churn more often in this data |
| `payment_method` | Electronic check, Mailed check, Bank transfer, Credit card | Electronic-check payers churn more often |
| `tech_support` | Yes, No, No internet service | Included support is protective |
| `online_security` | Yes, No, No internet service | Included security is protective |

The sample dataset contains nine more columns (gender, partner, streaming TV and
so on). They have no effect on churn in this data, and `total_charges` is just
`monthly_charges × tenure_months`, so the model does not use them.

---

## 👥 Collective prediction (CSV upload)

In the dashboard's **Collective prediction** tab, upload a CSV with one row per
customer:

- **Required:** `customer_id` and the eight input features above
- **Optional:** `customer_name`, `mobile_number`, `email`, and any other columns —
  they are shown in the results and never used by the model

Column names and category values are not case-sensitive. Rows with missing or
invalid values are listed separately with the reason, rather than guessed.
A ready-made example is in `data/sample_customers.csv`, and the app offers a
template download.

The results are sorted by churn probability, labelled High (70%+), Medium
(40–70%) or Low risk, and include each customer's main reasons. A slider sets
the probability at which a customer counts as "likely to churn". Both the
flagged customers and the full results can be downloaded as CSV. Files of up to
200,000 rows are supported; 100,000 rows take a few seconds.

---

## 🗄 Using a real SQL database instead of the sample CSV

`src/preprocessing.py` includes a `load_from_sql()` helper built on SQLAlchemy.
Point it at your customer database instead of the CSV:

```python
from preprocessing import load_from_sql

query = """
    SELECT customer_id, tenure_months, monthly_charges,
           num_support_calls, contract, internet_service,
           payment_method, tech_support, online_security, churn
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
| Upload says columns are missing | Compare your headers with the template CSV in the Collective prediction tab |
