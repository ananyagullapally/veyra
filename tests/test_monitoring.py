from unittest.mock import MagicMock, patch

from veyra.monitoring.metrics import get_prediction_metrics


@patch("veyra.monitoring.metrics.get_connection")
def test_prediction_metrics(mock_get_connection):
    cursor = MagicMock()
    cursor.fetchone.return_value = (6, 4, 2, 0.592393)

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    mock_get_connection.return_value.__enter__.return_value = connection

    metrics = get_prediction_metrics()

    assert metrics["total_predictions"] == 6
    assert metrics["flagged"] == 4
    assert metrics["not_flagged"] == 2
    assert metrics["flagged_rate"] == 4 / 6
    assert metrics["avg_fraud_probability"] == 0.592393


@patch("veyra.monitoring.metrics.get_connection")
def test_prediction_metrics_empty_database(mock_get_connection):
    cursor = MagicMock()
    cursor.fetchone.return_value = (0, 0, 0, None)

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor

    mock_get_connection.return_value.__enter__.return_value = connection

    metrics = get_prediction_metrics()

    assert metrics["total_predictions"] == 0
    assert metrics["flagged_rate"] == 0.0
    assert metrics["avg_fraud_probability"] == 0.0

@patch("veyra.monitoring.metrics.get_connection")
def test_metrics_query_recent_production_only(mock_get_connection):
    cursor = MagicMock()
    cursor.fetchone.return_value = (6, 4, 2, 0.592393)

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    mock_get_connection.return_value.__enter__.return_value = connection

    metrics = get_prediction_metrics()

    sql = cursor.execute.call_args.args[0]
    parameters = cursor.execute.call_args.args[1]
    normalized_sql = " ".join(sql.lower().split())

    assert "source = 'production'" in normalized_sql
    assert "predicted_at >= now() - (%s * interval '1 hour')" in normalized_sql
    assert parameters == (24,)

    assert metrics["total_predictions"] == 6
    assert metrics["flagged"] == 4
    assert metrics["not_flagged"] == 2
    assert metrics["flagged_rate"] == 4 / 6
    assert metrics["avg_fraud_probability"] == 0.592393

@patch("veyra.monitoring.config.MONITORING_WINDOW_HOURS", 48)
@patch("veyra.monitoring.metrics.get_connection")
def test_metrics_custom_monitoring_window(mock_get_connection):
    cursor = MagicMock()
    cursor.fetchone.return_value = (6, 4, 2, 0.592393)

    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    mock_get_connection.return_value.__enter__.return_value = connection

    metrics = get_prediction_metrics()

    sql, parameters = cursor.execute.call_args.args

    assert "source = 'production'" in sql.lower()
    assert parameters == (48,)
    assert metrics["total_predictions"] == 6
    assert metrics["flagged_rate"] == 4 / 6