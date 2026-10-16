
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from veyra.storage.consumer import run_storage_consumer


PREDICTION = {
    "application_id": "consumer-test-001",
    "fraud_probability": 0.85,
    "decision": "flagged",
    "threshold": 0.01320815272629261,
    "model": "veyra-fraud-detector",
    "alias": "champion",
}


def make_consumer():
    consumer = MagicMock()
    message = SimpleNamespace(
        value=json.dumps(PREDICTION).encode("utf-8")
    )
    consumer.__iter__.return_value = iter([message])
    return consumer


def test_commit_after_successful_storage():
    consumer = make_consumer()

    with patch(
        "veyra.storage.consumer.KafkaConsumer",
        return_value=consumer,
    ):
        with patch(
            "veyra.storage.consumer.save_prediction_with_retry"
        ) as save:
            run_storage_consumer()

    save.assert_called_once_with(PREDICTION)
    consumer.commit.assert_called_once()
    consumer.close.assert_called_once()


def test_no_commit_when_storage_fails():
    consumer = make_consumer()

    with patch(
        "veyra.storage.consumer.KafkaConsumer",
        return_value=consumer,
    ):
        with patch(
            "veyra.storage.consumer.save_prediction_with_retry",
            side_effect=RuntimeError("Database unavailable"),
        ):
            with pytest.raises(RuntimeError, match="Database unavailable"):
                run_storage_consumer()

    consumer.commit.assert_not_called()
    consumer.close.assert_called_once()


def test_no_commit_for_invalid_json():
    consumer = MagicMock()
    consumer.__iter__.return_value = iter([
        SimpleNamespace(value=b"invalid-json")
    ])

    with patch(
        "veyra.storage.consumer.KafkaConsumer",
        return_value=consumer,
    ):
        with patch(
            "veyra.storage.consumer.save_prediction_with_retry"
        ) as save:
            with pytest.raises(json.JSONDecodeError):
                run_storage_consumer()

    save.assert_not_called()
    consumer.commit.assert_not_called()
    consumer.close.assert_called_once()
