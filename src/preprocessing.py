"""
preprocessing.py
-----------------
Handles everything needed to turn the raw customer_churn.csv into
clean, model-ready arrays:

    1. Load data (CSV today, could be a SQL query tomorrow - see load_from_sql)
    2. Handle missing values
    3. Encode categorical columns
    4. Scale numeric columns
    5. Split into train / test sets

Beginner note: we wrap the encoders/scalers in a scikit-learn
ColumnTransformer + Pipeline so that the *exact same* transformations
are applied at training time and at prediction time (no code duplication,
no "it worked in the notebook but broke in the API" bugs).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_COL = "churn"
ID_COL = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "tech_support",
    "streaming_tv",
    "contract",
    "paperless_billing",
    "payment_method",
]


def load_data(csv_path: str = "data/customer_churn.csv") -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    return df


def load_from_sql(connection_string: str, query: str) -> pd.DataFrame:
    """
    Example helper for loading data from a SQL database instead of a CSV.
    Requires SQLAlchemy + the relevant DB driver (e.g. psycopg2, pyodbc).

    Example:
        query = '''
            SELECT customer_id, gender, senior_citizen, partner, dependents,
                   tenure_months, phone_service, multiple_lines,
                   internet_service, online_security, tech_support,
                   streaming_tv, contract, paperless_billing, payment_method,
                   monthly_charges, total_charges, num_support_calls, churn
            FROM customers
        '''
        df = load_from_sql("postgresql://user:pass@host:5432/db", query)
    """
    from sqlalchemy import create_engine

    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates, fix dtypes, trim whitespace."""
    df = df.copy()
    df = df.drop_duplicates()

    # Make sure numeric columns are actually numeric (SQL exports / CSVs
    # sometimes bring these in as strings with stray spaces).
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace from string/categorical columns
    for col in CATEGORICAL_FEATURES + [TARGET_COL]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
      - imputes + scales numeric columns
      - imputes + one-hot encodes categorical columns

    Returning the *unfitted* transformer lets train_model.py fit it only
    on the training split, avoiding data leakage from the test set.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Returns human-readable feature names after the ColumnTransformer runs.
    Needed later for feature-importance plots and SHAP."""
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + cat_names


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Splits the cleaned dataframe into X_train, X_test, y_train, y_test.
    y is encoded as 0/1 (No/Yes) here, since scikit-learn models expect numbers.
    """
    df = clean_data(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET_COL] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick manual test: run `python src/preprocessing.py` from the project root
    df = load_data()
    df = clean_data(df)
    print("Rows after cleaning:", len(df))
    print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Churn rate (train):", y_train.mean().round(3))
