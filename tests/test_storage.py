from unittest.mock import MagicMock, patch

import pytest

from veyra.storage.predictions import save_prediction


@pytest.fixture
def prediction():
    return {
        "application_id": "test-001",
        "fraud_probability": 0.92,
        "decision": "flagged",
        "threshold": 0.01320815272629261,
        "model": "veyra-fraud-detector",
        "alias": "champion",
    }


def test_save_prediction_executes_upsert(prediction):
    connection = MagicMock()
    cursor = MagicMock()

    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor

    with patch(
        "veyra.storage.predictions.get_connection",
        return_value=connection,
    ):
        save_prediction(prediction)

    cursor.execute.assert_called_once()

    query, params = cursor.execute.call_args.args

    assert "ON CONFLICT (application_id)" in query
    assert "DO UPDATE SET" in query
    assert params == prediction


def test_save_prediction_database_failure(prediction):
    connection = MagicMock()
    cursor = MagicMock()

    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.execute.side_effect = RuntimeError("Database unavailable")

    with patch(
        "veyra.storage.predictions.get_connection",
        return_value=connection,
    ):
        with pytest.raises(RuntimeError, match="Database unavailable"):
            save_prediction(prediction)


def test_save_prediction_connection_failure(prediction):
    with patch(
        "veyra.storage.predictions.get_connection",
        side_effect=ConnectionError("PostgreSQL unavailable"),
    ):
        with pytest.raises(ConnectionError):
            save_prediction(prediction)