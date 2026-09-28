from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/Base.csv")
PROCESSED_DATA_DIR = Path("data/processed")

APPLICATIONS_PATH = PROCESSED_DATA_DIR / "applications.parquet"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.parquet"

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

    train_df, validation_df, test_df = split_temporally(df)

    train_df, validation_df, test_df = add_split_column(
        train_df,
        validation_df,
        test_df,
    )

    train_applications, train_labels = separate_features_and_labels(train_df)

    validation_applications, validation_labels = separate_features_and_labels(
        validation_df
    )

    test_applications, test_labels = separate_features_and_labels(test_df)

    applications_df = pd.concat(
        [
            train_applications,
            validation_applications,
            test_applications,
        ],
        ignore_index=True,
    )

    labels_df = pd.concat(
        [
            train_labels,
            validation_labels,
            test_labels,
        ],
        ignore_index=True,
    )

    save_processed_data(applications_df, labels_df)
    validate_saved_data()

def validate_saved_data() -> None:
    """Read processed files back and verify their basic integrity."""

    applications_df = pd.read_parquet(APPLICATIONS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    print("\nSaved data validation:")
    print(f"Applications: {len(applications_df):,} rows")
    print(f"Labels:       {len(labels_df):,} rows")

    print(
        f"fraud_bool in applications: "
        f"{'fraud_bool' in applications_df.columns}"
    )
    print(
        f"fraud_bool in labels: "
        f"{'fraud_bool' in labels_df.columns}"
    )

    print("\nApplication splits:")
    print(applications_df["dataset_split"].value_counts())

    print("\nLabel splits:")
    print(labels_df["dataset_split"].value_counts())

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

def split_temporally(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split applications chronologically using the BAF month field."""

    train_df = df[df["month"].between(0, 5)].copy()
    validation_df = df[df["month"] == 6].copy()
    test_df = df[df["month"] == 7].copy()

    print("\nTemporal split created:")
    print(f"Train:      {len(train_df):,} applications")
    print(f"Validation: {len(validation_df):,} applications")
    print(f"Test:       {len(test_df):,} applications")

    return train_df, validation_df, test_df

def add_split_column(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label each application with its dataset split."""

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["dataset_split"] = "train"
    validation_df["dataset_split"] = "validation"
    test_df["dataset_split"] = "test"

    return train_df, validation_df, test_df

def separate_features_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate application data from the fraud label."""

    label_columns = [
        "application_id",
        "fraud_bool",
        "month",
        "dataset_split",
    ]

    labels_df = df[label_columns].copy()

    applications_df = df.drop(columns=["fraud_bool"]).copy()

    return applications_df, labels_df

def save_processed_data(
    applications_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> None:
    """Save prepared application data and fraud labels as Parquet files."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    applications_df.to_parquet(
        APPLICATIONS_PATH,
        index=False,
    )

    labels_df.to_parquet(
        LABELS_PATH,
        index=False,
    )

    print("\nProcessed data saved:")
    print(f"Applications: {APPLICATIONS_PATH}")
    print(f"Labels:       {LABELS_PATH}")

if __name__ == "__main__":
    main()