from veyra.features.preprocessing import build_preprocessor
from pathlib import Path

import pandas as pd


APPLICATIONS_PATH = Path("data/processed/applications.parquet")
LABELS_PATH = Path("data/processed/labels.parquet")


def load_training_data() -> pd.DataFrame:
    """Load and join training applications with their fraud labels."""

    applications_df = pd.read_parquet(APPLICATIONS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    train_applications = applications_df[
        applications_df["dataset_split"] == "train"
    ].copy()

    train_labels = labels_df[
        labels_df["dataset_split"] == "train"
    ][["application_id", "fraud_bool"]].copy()

    train_df = train_applications.merge(
        train_labels,
        on="application_id",
        how="inner",
        validate="one_to_one",
    )

    print("Training data loaded:")
    print(f"Applications: {len(train_df):,}")
    print(f"Columns:      {len(train_df.columns)}")
    print(f"Fraud cases:  {train_df['fraud_bool'].sum():,}")
    print(f"Fraud rate:   {train_df['fraud_bool'].mean():.4%}")

    return train_df

def load_split_data(split_name: str) -> pd.DataFrame:
    """Load applications and labels for a requested dataset split."""

    valid_splits = {"train", "validation", "test"}

    if split_name not in valid_splits:
        raise ValueError(
            f"Invalid split '{split_name}'. "
            f"Expected one of {sorted(valid_splits)}."
        )

    applications_df = pd.read_parquet(APPLICATIONS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    split_applications = applications_df[
        applications_df["dataset_split"] == split_name
    ].copy()

    split_labels = labels_df[
        labels_df["dataset_split"] == split_name
    ][["application_id", "fraud_bool"]].copy()

    split_df = split_applications.merge(
        split_labels,
        on="application_id",
        how="inner",
        validate="one_to_one",
    )

    return split_df

def create_feature_target_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the fraud target."""

    excluded_columns = [
        "application_id",
        "month",
        "dataset_split",
        "fraud_bool",
    ]

    X = df.drop(columns=excluded_columns).copy()
    y = df["fraud_bool"].copy()

    print("\nFeature/target split:")
    print(f"Features: {X.shape[1]}")
    print(f"Rows:     {X.shape[0]:,}")
    print(f"Target:   fraud_bool")

    return X, y

def inspect_features(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Inspect feature types and missing values before preprocessing."""

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("\nFeature inspection:")
    print(f"Numeric features:     {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    print("\nCategorical columns:")
    for column in categorical_columns:
        print(f"  - {column}")

    print("\nFeatures with missing values:")
    missing_counts = X.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    for column, count in missing_counts.items():
        print(f"  - {column}: {count:,}")

    return numeric_columns, categorical_columns

def main():
    train_df = load_training_data()

    X_train, y_train = create_feature_target_split(train_df)

    numeric_columns, categorical_columns = inspect_features(X_train)

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    X_train_transformed = preprocessor.fit_transform(X_train)

    print("\nPreprocessing complete:")
    print(f"Raw feature shape:         {X_train.shape}")
    print(f"Transformed feature shape: {X_train_transformed.shape}")
    print(f"Target shape:              {y_train.shape}")

    validation_df = load_split_data("validation")

    X_validation, y_validation = create_feature_target_split(
        validation_df
    )

    print("\nValidation data loaded:")
    print(f"Applications: {len(validation_df):,}")
    print(f"Features:     {X_validation.shape}")
    print(f"Target:       {y_validation.shape}")
    print(f"Fraud cases:  {y_validation.sum():,}")
    print(f"Fraud rate:   {y_validation.mean():.4%}")

if __name__ == "__main__":
    main()