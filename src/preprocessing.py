"""
preprocessing.py
-----------------
Handles everything needed to turn raw customer data into clean,
model-ready input:

    1. Load data (CSV today, could be a SQL query tomorrow - see load_from_sql)
    2. Clean it (duplicates, dtypes, whitespace)
    3. Impute, scale and encode inside a scikit-learn ColumnTransformer
    4. Split into train / test sets
    5. Validate a customer file uploaded for batch prediction

Beginner note: the encoders/scalers live inside a Pipeline together with
the model, so the *exact same* transformations run at training time and
at prediction time. Callers only ever pass raw values.

Feature choice: churn in this dataset is driven by exactly the eight
features below (see data/generate_sample_data.py). The remaining columns
in the CSV have no effect on churn, and total_charges is just
monthly_charges x tenure_months, so none of them are used by the model.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "churn"
ID_COL = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "num_support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
    "tech_support",
    "online_security",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Every value the model was trained on, per categorical feature.
ALLOWED_VALUES = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "tech_support": ["Yes", "No", "No internet service"],
    "online_security": ["Yes", "No", "No internet service"],
}

# Valid range for each numeric feature: (min, max). None means unbounded.
NUMERIC_RANGES = {
    "tenure_months": (0, None),
    "monthly_charges": (0, None),
    "num_support_calls": (0, None),
}

# Human-readable names, used in the dashboard and in explanations.
FEATURE_LABELS = {
    "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges",
    "num_support_calls": "Support calls",
    "contract": "Contract",
    "internet_service": "Internet service",
    "payment_method": "Payment method",
    "tech_support": "Tech support",
    "online_security": "Online security",
}

# Identity columns for batch files. Only customer_id is required; these
# and any other extra columns are carried through to the results untouched
# and are never seen by the model.
OPTIONAL_ID_COLUMNS = ["customer_name", "mobile_number", "email"]


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
            SELECT customer_id, tenure_months, monthly_charges,
                   num_support_calls, contract, internet_service,
                   payment_method, tech_support, online_security, churn
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

    X = df[MODEL_FEATURES]
    y = (df[TARGET_COL] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _normalise_column_name(name) -> str:
    """'Monthly Charges ' -> 'monthly_charges'."""
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def template_dataframe() -> pd.DataFrame:
    """A small example of the batch upload format, for users to download."""
    return pd.DataFrame(
        [
            {
                "customer_id": "CUST0001",
                "customer_name": "Example Customer A",
                "mobile_number": "+91-90000-00001",
                "email": "customer.a@example.com",
                "tenure_months": 8,
                "monthly_charges": 85.0,
                "num_support_calls": 3,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "tech_support": "Yes",
                "online_security": "No",
            },
            {
                "customer_id": "CUST0002",
                "customer_name": "Example Customer B",
                "mobile_number": "+91-90000-00002",
                "email": "customer.b@example.com",
                "tenure_months": 48,
                "monthly_charges": 55.0,
                "num_support_calls": 0,
                "contract": "Two year",
                "internet_service": "DSL",
                "payment_method": "Bank transfer",
                "tech_support": "Yes",
                "online_security": "Yes",
            },
        ]
    )


def validate_batch(raw: pd.DataFrame):
    """
    Checks an uploaded customer file and splits it into rows the model can
    score and rows it cannot.

    Returns (valid_df, rejected_df, missing_columns):
      - missing_columns: required columns absent from the file. If this is
        non-empty the file cannot be scored at all, and the other two
        return values are empty.
      - valid_df: rows ready for the model. Column names are normalised,
        categorical values are converted to their canonical spelling, and
        every original column is kept in its original order.
      - rejected_df: rows that failed validation, with a 'problem' column
        explaining why, so the user can fix and re-upload them.

    Invalid rows are rejected rather than silently imputed: a guessed
    value would produce a confident-looking prediction that is wrong.
    """
    df = raw.copy()
    df.columns = [_normalise_column_name(c) for c in df.columns]

    required = [ID_COL] + MODEL_FEATURES
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        return pd.DataFrame(), pd.DataFrame(), missing_columns

    problems = pd.Series([[] for _ in range(len(df))], index=df.index)

    def flag(mask, message):
        for idx in df.index[mask]:
            problems[idx].append(message)

    ids = df[ID_COL].astype("string").str.strip()
    flag(ids.isna() | (ids == ""), "missing customer_id")
    df[ID_COL] = ids

    for col in NUMERIC_FEATURES:
        original = df[col]
        numbers = pd.to_numeric(original, errors="coerce")
        blank = original.isna() | (original.astype("string").str.strip() == "")
        flag(blank, f"missing {col}")
        flag(numbers.isna() & ~blank, f"{col} is not a number")
        low, high = NUMERIC_RANGES[col]
        if low is not None:
            flag(numbers < low, f"{col} cannot be below {low}")
        if high is not None:
            flag(numbers > high, f"{col} cannot be above {high}")
        df[col] = numbers

    for col, allowed in ALLOWED_VALUES.items():
        # Accept any capitalisation, but store the spelling the model knows.
        lookup = {value.lower(): value for value in allowed}
        text = df[col].astype("string").str.strip()
        canonical = text.str.lower().map(lookup)
        blank = text.isna() | (text == "")
        flag(blank, f"missing {col}")
        flag(canonical.isna() & ~blank, f"{col} must be one of: {', '.join(allowed)}")
        df[col] = canonical

    has_problem = problems.map(bool)
    valid_df = df[~has_problem].copy()
    rejected_df = raw[has_problem.to_numpy()].copy()
    rejected_df.insert(0, "problem", problems[has_problem].map("; ".join).to_numpy())
    return valid_df, rejected_df, []


if __name__ == "__main__":
    # Quick manual test: run `python src/preprocessing.py` from the project root
    df = load_data()
    df = clean_data(df)
    print("Rows after cleaning:", len(df))
    print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Churn rate (train):", y_train.mean().round(3))
