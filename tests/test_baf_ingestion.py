import pandas as pd
import pytest

from veyra.ingestion.prepare_baf import (
    REQUIRED_COLUMNS,
    add_application_ids,
    add_split_column,
    handle_sentinel_missingness,
    remove_constant_features,
    save_processed_data,
    separate_features_and_labels,
    split_temporally,
    validate_schema,
)

def test_temporal_split_uses_correct_months():
    df = pd.DataFrame(
        {
            "application_id": range(1, 9),
            "month": range(8),
        }
    )

    train_df, validation_df, test_df = split_temporally(df)

    assert set(train_df["month"].unique()) == {0, 1, 2, 3, 4, 5}
    assert set(validation_df["month"].unique()) == {6}
    assert set(test_df["month"].unique()) == {7}


def test_temporal_split_preserves_all_applications():
    df = pd.DataFrame(
        {
            "application_id": range(1, 9),
            "month": range(8),
        }
    )

    train_df, validation_df, test_df = split_temporally(df)

    total_rows = len(train_df) + len(validation_df) + len(test_df)

    assert total_rows == len(df)


def test_temporal_split_has_no_application_overlap():
    df = pd.DataFrame(
        {
            "application_id": range(1, 9),
            "month": range(8),
        }
    )

    train_df, validation_df, test_df = split_temporally(df)

    train_ids = set(train_df["application_id"])
    validation_ids = set(validation_df["application_id"])
    test_ids = set(test_df["application_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

def make_valid_baf_df() -> pd.DataFrame:
    """Create a small BAF-like DataFrame for testing."""

    data = {}

    for column in REQUIRED_COLUMNS:
        data[column] = [0, 0, 0]

    # Valid fraud labels
    data["fraud_bool"] = [0, 1, 0]

    # Sentinel-value columns
    data["prev_address_months_count"] = [-1, 12, 24]
    data["current_address_months_count"] = [6, -1, 18]
    data["bank_months_count"] = [-1, 10, 20]
    data["device_distinct_emails_8w"] = [1, 2, -1]

    # Constant feature found in BAF
    data["device_fraud_count"] = [0, 0, 0]

    return pd.DataFrame(data)


def test_validate_schema_accepts_valid_data():
    df = make_valid_baf_df()

    validate_schema(df)


def test_validate_schema_rejects_missing_column():
    df = make_valid_baf_df()

    df = df.drop(columns=["income"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df)


def test_validate_schema_rejects_invalid_fraud_label():
    df = make_valid_baf_df()

    df.loc[1, "fraud_bool"] = 7

    with pytest.raises(ValueError, match="fraud_bool must contain only 0 and 1"):
        validate_schema(df)


def test_application_ids_are_created():
    df = make_valid_baf_df()

    result = add_application_ids(df)

    assert result["application_id"].tolist() == [1, 2, 3]


def test_sentinel_missingness_is_handled():
    df = make_valid_baf_df()

    result = handle_sentinel_missingness(df)

    assert pd.isna(result.loc[0, "prev_address_months_count"])
    assert result.loc[0, "prev_address_months_count_missing"] == 1
    assert result.loc[1, "prev_address_months_count_missing"] == 0

    assert pd.isna(result.loc[1, "current_address_months_count"])
    assert pd.isna(result.loc[0, "bank_months_count"])
    assert pd.isna(result.loc[2, "device_distinct_emails_8w"])


def test_constant_feature_is_removed():
    df = make_valid_baf_df()

    result = remove_constant_features(df)

    assert "device_fraud_count" not in result.columns

def test_split_column_is_added_correctly():
    df = pd.DataFrame(
        {
            "application_id": range(1, 9),
            "month": range(8),
        }
    )

    train_df, validation_df, test_df = split_temporally(df)

    train_df, validation_df, test_df = add_split_column(
        train_df,
        validation_df,
        test_df,
    )

    assert set(train_df["dataset_split"].unique()) == {"train"}
    assert set(validation_df["dataset_split"].unique()) == {"validation"}
    assert set(test_df["dataset_split"].unique()) == {"test"}

def test_features_and_labels_are_separated():
    df = make_valid_baf_df()
    df = add_application_ids(df)
    df["dataset_split"] = "train"

    applications_df, labels_df = separate_features_and_labels(df)

    # The application stream must never contain the target.
    assert "fraud_bool" not in applications_df.columns

    # The labels table must retain the target.
    assert "fraud_bool" in labels_df.columns

    # IDs must remain available so predictions can later be matched to labels.
    assert "application_id" in applications_df.columns
    assert "application_id" in labels_df.columns

    # Separation must not lose any applications.
    assert len(applications_df) == len(df)
    assert len(labels_df) == len(df)

def test_processed_data_can_be_saved_and_read_back(tmp_path, monkeypatch):
    applications_path = tmp_path / "applications.parquet"
    labels_path = tmp_path / "labels.parquet"

    monkeypatch.setattr(
        "veyra.ingestion.prepare_baf.PROCESSED_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "veyra.ingestion.prepare_baf.APPLICATIONS_PATH",
        applications_path,
    )
    monkeypatch.setattr(
        "veyra.ingestion.prepare_baf.LABELS_PATH",
        labels_path,
    )

    applications_df = pd.DataFrame(
        {
            "application_id": [1, 2],
            "income": [0.5, 0.8],
            "dataset_split": ["train", "test"],
        }
    )

    labels_df = pd.DataFrame(
        {
            "application_id": [1, 2],
            "fraud_bool": [0, 1],
            "dataset_split": ["train", "test"],
        }
    )

    save_processed_data(applications_df, labels_df)

    saved_applications = pd.read_parquet(applications_path)
    saved_labels = pd.read_parquet(labels_path)

    assert len(saved_applications) == 2
    assert len(saved_labels) == 2
    assert "fraud_bool" not in saved_applications.columns
    assert "fraud_bool" in saved_labels.columns