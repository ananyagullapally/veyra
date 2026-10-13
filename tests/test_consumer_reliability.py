
from unittest.mock import MagicMock, patch

import pytest
import json
from veyra.streaming.consumer import publish_prediction
from veyra.streaming.dlq import publish_to_dlq
from veyra.streaming.consumer import process_application_message
from veyra.streaming.errors import PermanentProcessingError

def test_prediction_delivery_succeeds():
    producer = MagicMock()
    producer.flush.return_value = 0

    result = publish_prediction(
        producer,
        "app-123",
        {"fraud_probability": 0.85},
    )

    assert result["application_id"] == "app-123"
    producer.produce.assert_called_once()


def test_prediction_delivery_failure_raises():
    producer = MagicMock()
    producer.flush.return_value = 1

    with pytest.raises(RuntimeError, match="Prediction delivery failed"):
        publish_prediction(
            producer,
            "app-123",
            {"fraud_probability": 0.85},
        )


def test_dlq_delivery_failure_raises():
    producer = MagicMock()
    producer.flush.return_value = 1

    with pytest.raises(RuntimeError, match="DLQ delivery failed"):
        publish_to_dlq(
            producer=producer,
            application_id="app-123",
            original_payload={"features": {}},
            error=ValueError("Invalid event"),
            attempt_count=1,
        )


def make_message(event):
    message = MagicMock()
    message.value.return_value = json.dumps(event).encode("utf-8")
    message.key.return_value = b"app-123"
    return message


def test_consumer_commits_after_prediction():
    consumer = MagicMock()
    producer = MagicMock()
    message = make_message({
        "application_id": "app-123",
        "features": {"income": 50000},
    })

    with patch(
        "veyra.streaming.consumer.score_with_recovery",
        return_value={"fraud_probability": 0.85},
    ), patch(
        "veyra.streaming.consumer.publish_prediction",
        return_value={"application_id": "app-123"},
    ):
        result = process_application_message(message, consumer, producer)

    assert result == "predicted"
    consumer.commit.assert_called_once_with(
        message=message, asynchronous=False
    )


def test_consumer_does_not_commit_if_prediction_publish_fails():
    consumer = MagicMock()
    producer = MagicMock()
    message = make_message({
        "application_id": "app-123",
        "features": {"income": 50000},
    })

    with patch(
        "veyra.streaming.consumer.score_with_recovery",
        return_value={"fraud_probability": 0.85},
    ), patch(
        "veyra.streaming.consumer.publish_prediction",
        side_effect=RuntimeError("Kafka delivery failed"),
    ):
        with pytest.raises(RuntimeError):
            process_application_message(message, consumer, producer)

    consumer.commit.assert_not_called()


def test_consumer_commits_after_dlq():
    consumer = MagicMock()
    producer = MagicMock()
    message = make_message({
        "application_id": "app-123",
        "features": {"income": -1},
    })

    with patch(
        "veyra.streaming.consumer.score_with_recovery",
        side_effect=PermanentProcessingError("Invalid features"),
    ), patch(
        "veyra.streaming.consumer.publish_to_dlq"
    ):
        result = process_application_message(message, consumer, producer)

    assert result == "dlq"
    consumer.commit.assert_called_once_with(
        message=message, asynchronous=False
    )


def test_consumer_does_not_commit_if_dlq_publish_fails():
    consumer = MagicMock()
    producer = MagicMock()
    message = make_message({
        "application_id": "app-123",
        "features": {"income": -1},
    })

    with patch(
        "veyra.streaming.consumer.score_with_recovery",
        side_effect=PermanentProcessingError("Invalid features"),
    ), patch(
        "veyra.streaming.consumer.publish_to_dlq",
        side_effect=RuntimeError("DLQ delivery failed"),
    ):
        with pytest.raises(RuntimeError):
            process_application_message(message, consumer, producer)

    consumer.commit.assert_not_called()
