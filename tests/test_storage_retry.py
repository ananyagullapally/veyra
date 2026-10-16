from unittest.mock import patch

import psycopg
import pytest

from veyra.storage.retry import save_prediction_with_retry


PREDICTION = {
    "application_id": "retry-test-001",
    "fraud_probability": 0.85,
    "decision": "flagged",
    "threshold": 0.01320815272629261,
    "model": "veyra-fraud-detector",
    "alias": "champion",
}


def test_success_without_retry():
    with patch("veyra.storage.retry.save_prediction") as save:
        with patch("veyra.storage.retry.time.sleep") as sleep:
            save_prediction_with_retry(PREDICTION)

    save.assert_called_once_with(PREDICTION)
    sleep.assert_not_called()


def test_retry_then_success():
    with patch("veyra.storage.retry.save_prediction") as save:
        with patch("veyra.storage.retry.time.sleep") as sleep:
            save.side_effect = [
                psycopg.OperationalError("Connection unavailable"),
                psycopg.OperationalError("Connection unavailable"),
                None,
            ]

            save_prediction_with_retry(PREDICTION)

    assert save.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]


def test_failure_after_max_attempts():
    with patch("veyra.storage.retry.save_prediction") as save:
        with patch("veyra.storage.retry.time.sleep") as sleep:
            save.side_effect = psycopg.OperationalError(
                "Database unavailable"
            )

            with pytest.raises(psycopg.OperationalError):
                save_prediction_with_retry(PREDICTION)

    assert save.call_count == 3
    assert sleep.call_count == 2


def test_non_retryable_error():
    with patch("veyra.storage.retry.save_prediction") as save:
        with patch("veyra.storage.retry.time.sleep") as sleep:
            save.side_effect = psycopg.errors.CheckViolation(
                "Invalid prediction"
            )

            with pytest.raises(psycopg.errors.CheckViolation):
                save_prediction_with_retry(PREDICTION)

    save.assert_called_once()
    sleep.assert_not_called()


def test_invalid_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        save_prediction_with_retry(PREDICTION, max_attempts=0)