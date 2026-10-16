from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from veyra.monitoring.drift_monitor import check_prediction_drift


def mock_database(mock_get_connection, probabilities):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (float(probability),) for probability in probabilities
    ]

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    mock_get_connection.return_value.__enter__.return_value = connection


@patch("veyra.monitoring.drift_monitor.get_connection")
@patch("veyra.monitoring.drift_monitor.pd.read_parquet")
def test_insufficient_data(mock_read_parquet, mock_get_connection):
    mock_read_parquet.return_value = pd.DataFrame({
        "fraud_probability": [0.01, 0.02, 0.03]
    })
    mock_database(mock_get_connection, [0.1] * 6)

    result = check_prediction_drift()

    assert result["status"] == "insufficient_data"
    assert result["sample_count"] == 6
    assert result["psi"] is None


@patch("veyra.monitoring.drift_monitor.get_connection")
@patch("veyra.monitoring.drift_monitor.pd.read_parquet")
def test_stable_distribution(mock_read_parquet, mock_get_connection):
    reference = np.linspace(0.001, 0.999, 1000)

    mock_read_parquet.return_value = pd.DataFrame({
        "fraud_probability": reference
    })
    mock_database(mock_get_connection, reference)

    result = check_prediction_drift()

    assert result["status"] == "stable"
    assert result["sample_count"] == 1000
    assert result["psi"] < 0.10


@patch("veyra.monitoring.drift_monitor.get_connection")
@patch("veyra.monitoring.drift_monitor.pd.read_parquet")
def test_significant_drift(mock_read_parquet, mock_get_connection):
    rng = np.random.default_rng(42)

    reference = rng.beta(1, 20, 10000)
    shifted = rng.beta(2, 12, 1000)

    mock_read_parquet.return_value = pd.DataFrame({
        "fraud_probability": reference
    })
    mock_database(mock_get_connection, shifted)

    result = check_prediction_drift()

    assert result["status"] == "significant_drift"
    assert result["sample_count"] == 1000
    assert result["psi"] > 0.25

@patch("veyra.monitoring.drift_monitor.get_connection")
@patch("veyra.monitoring.drift_monitor.pd.read_parquet")
def test_queries_recent_production_predictions(
    mock_read_parquet, mock_get_connection
):
    mock_read_parquet.return_value = pd.DataFrame({
        "fraud_probability": [0.01, 0.02, 0.03]
    })

    mock_database(mock_get_connection, [0.1] * 6)

    result = check_prediction_drift()

    connection = mock_get_connection.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value

    sql = cursor.execute.call_args.args[0]
    normalized_sql = " ".join(sql.lower().split())

    assert "source = 'production'" in normalized_sql
    assert "predicted_at >= now() - (%s * interval '1 hour')" in normalized_sql
    assert cursor.execute.call_args.args[1] == (24,)

    assert result["status"] == "insufficient_data"
    assert result["sample_count"] == 6
    assert result["minimum_required"] == 100

@patch("veyra.monitoring.config.MONITORING_WINDOW_HOURS", 48)
@patch("veyra.monitoring.drift_monitor.get_connection")
@patch("veyra.monitoring.drift_monitor.pd.read_parquet")
def test_custom_monitoring_window(mock_read_parquet, mock_get_connection):
    mock_read_parquet.return_value = pd.DataFrame({
        "fraud_probability": [0.01, 0.02, 0.03]
    })

    mock_database(mock_get_connection, [0.1] * 6)

    result = check_prediction_drift()

    connection = mock_get_connection.return_value.__enter__.return_value
    cursor = connection.cursor.return_value.__enter__.return_value

    assert cursor.execute.call_args.args[1] == (48,)
    assert result["status"] == "insufficient_data"