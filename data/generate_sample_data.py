"""
generate_sample_data.py
------------------------
Creates a synthetic but realistic "telecom-style" customer churn dataset
and saves it as data/customer_churn.csv

This is only used so the project has a ready-to-use dataset out of the box.
In a real project you would replace this with your own data
(e.g. pulled from a SQL database - see the SQL note in README.md).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000  # number of customers

# --- Demographic / account features -----------------------------------
customer_id = [f"CUST{i:05d}" for i in range(1, N + 1)]
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N)
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

tenure_months = np.random.randint(0, 73, N)  # 0-72 months with the company

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20]
)
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.6, 0.4])

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20])
online_security = np.random.choice(["Yes", "No", "No internet service"], N)
tech_support = np.random.choice(["Yes", "No", "No internet service"], N)
streaming_tv = np.random.choice(["Yes", "No", "No internet service"], N)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.random.choice(["Yes", "No", "No phone service"], N)

monthly_charges = np.round(np.random.uniform(18, 120, N), 2)
total_charges = np.round(monthly_charges * tenure_months + np.random.uniform(0, 50, N), 2)

num_support_calls = np.random.poisson(1.5, N)

# --- Build churn probability from a hand-crafted "true" relationship ---
# This keeps the dataset useful for teaching feature importance / SHAP,
# since we know which features SHOULD matter.
churn_logit = (
    -1.5
    + 1.8 * (contract == "Month-to-month")
    - 1.0 * (contract == "Two year")
    + 0.015 * (monthly_charges - 60)
    - 0.03 * tenure_months
    + 0.9 * (internet_service == "Fiber optic")
    + 0.25 * num_support_calls
    + 0.4 * (payment_method == "Electronic check")
    - 0.3 * (tech_support == "Yes")
    - 0.3 * (online_security == "Yes")
    + np.random.normal(0, 0.6, N)  # noise
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = np.where(churn_prob > np.random.uniform(0, 1, N), "Yes", "No")

df = pd.DataFrame(
    {
        "customer_id": customer_id,
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
        "churn": churn,
    }
)

# Inject a few missing values so preprocessing.py has something real to clean
missing_idx = np.random.choice(df.index, 40, replace=False)
df.loc[missing_idx, "total_charges"] = np.nan

out_path = "data/customer_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["churn"].value_counts(normalize=True))
