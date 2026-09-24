import pandas as pd
import pytest

from veyra.ingestion.prepare_baf import (
    REQUIRED_COLUMNS,
    add_application_ids,
    handle_sentinel_missingness,
    remove_constant_features,
    validate_schema,
)


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