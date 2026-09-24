from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/Base.csv")


REQUIRED_COLUMNS = {
    "fraud_bool",
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
}

SENTINEL_MISSING_COLUMNS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "device_distinct_emails_8w",
]

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw BAF Base dataset."""

    print(f"Loading BAF from {path}...")

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} applications.")

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the expected structure of the BAF dataset."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset contains no applications.")

    if df["fraud_bool"].isna().any():
        raise ValueError("Fraud labels contain missing values.")

    if not set(df["fraud_bool"].unique()).issubset({0, 1}):
        raise ValueError("fraud_bool must contain only 0 and 1.")

    print("Schema validation passed.")


def main():
    df = load_data(RAW_DATA_PATH)

    validate_schema(df)

    df = add_application_ids(df)

    df = handle_sentinel_missingness(df)
    
    df = remove_constant_features(df)

    print("\nFinal preprocessing check:")
    print(f"Columns: {len(df.columns)}")
    print(
        f"device_fraud_count present: "
        f"{'device_fraud_count' in df.columns}"
    )

def add_application_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a stable unique identifier to every application."""

    df = df.copy()

    df.insert(
        0,
        "application_id",
        range(1, len(df) + 1),
    )

    print("Application IDs created.")

    return df

def handle_sentinel_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sentinel -1 values to missing values and preserve missingness indicators."""

    df = df.copy()

    for column in SENTINEL_MISSING_COLUMNS:
        missing_mask = df[column] == -1

        df[f"{column}_missing"] = missing_mask.astype("int8")

        df.loc[missing_mask, column] = pd.NA

        print(
            f"{column}: converted "
            f"{missing_mask.sum():,} sentinel values"
        )

    print("Sentinel missing values handled.")

    return df

def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove features containing only one unique value."""

    df = df.copy()

    protected_columns = {
        "application_id",
        "fraud_bool",
        "month",
    }

    constant_columns = [
        column
        for column in df.columns
        if column not in protected_columns
        and df[column].nunique(dropna=False) == 1
    ]

    if constant_columns:
        print(f"Constant features detected: {constant_columns}")

        df = df.drop(columns=constant_columns)

        print("Constant features removed.")
    else:
        print("No constant features detected.")

    return df

if __name__ == "__main__":
    main()