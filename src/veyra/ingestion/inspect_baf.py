from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/Base.csv")


def main():
    print("Loading BAF Base dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")

    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n=== DUPLICATE ROWS ===")
    print(f"Duplicates: {df.duplicated().sum():,}")

    print("\n=== FRAUD DISTRIBUTION ===")
    fraud_counts = df["fraud_bool"].value_counts().sort_index()
    print(fraud_counts)

    fraud_rate = df["fraud_bool"].mean() * 100
    print(f"Fraud rate: {fraud_rate:.4f}%")

    print("\n=== UNIQUE VALUES PER COLUMN ===")
    print(df.nunique().sort_values())

    print("\n=== POSSIBLE SENTINEL VALUES ===")
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        negative_count = (numeric_df[column] < 0).sum()

        if negative_count > 0:
            print(
                f"{column}: "
                f"{negative_count:,} negative values | "
                f"min = {numeric_df[column].min()}"
            )

    print("\n=== CATEGORICAL VALUES ===")
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

    print("\n=== MONTH DISTRIBUTION ===")
    print(df["month"].value_counts().sort_index())

    print("\n=== FRAUD BY MONTH ===")
    fraud_by_month = df.groupby("month")["fraud_bool"].agg(
        applications="count",
        fraud_cases="sum",
        fraud_rate="mean",
    )

    fraud_by_month["fraud_rate"] *= 100

    print(fraud_by_month)

    print("\n=== NUMERIC FEATURE RANGES ===")

    for column in numeric_df.columns:
        print(
            f"{column}: "
            f"min={df[column].min()}, "
            f"max={df[column].max()}"
        )

    print("\n=== CONSTANT FEATURES ===")

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) == 1
    ]

    print(constant_columns)

    for column in constant_columns:
        print(
            f"{column}: {df[column].iloc[0]}"
        )

if __name__ == "__main__":
    main()