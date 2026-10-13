import json
import httpx
import pytest

from veyra.streaming.consumer import parse_application_event
from veyra.streaming.errors import (
    PermanentProcessingError,
    TransientProcessingError,
    classify_http_error,
)
from veyra.streaming.recovery import score_with_recovery
from veyra.streaming.errors import (
    TransientProcessingError,
    PermanentProcessingError,
)
from unittest.mock import patch
from veyra.streaming.retry import retry_operation
from unittest.mock import Mock
from veyra.streaming.scoring import score_application
from unittest.mock import MagicMock
from veyra.streaming.dlq import publish_to_dlq

def test_valid_application_event():
    event = {
        "application_id": "123",
        "features": {"income": 50000},
    }

    result = parse_application_event(json.dumps(event).encode("utf-8"))

    assert result == event


def test_invalid_json_event():
    result = parse_application_event(b"not valid json")

    assert result is None


def test_missing_features_event():
    event = {"application_id": "123"}

    result = parse_application_event(json.dumps(event).encode("utf-8"))

    assert result is None


def test_features_must_be_dictionary():
    event = {
        "application_id": "123",
        "features": "invalid",
    }

    result = parse_application_event(json.dumps(event).encode("utf-8"))

    assert result is None


def test_json_array_is_rejected():
    result = parse_application_event(b'[1, 2, 3]')

    assert result is None

def test_timeout_is_transient():
    error = httpx.ReadTimeout("Request timed out")
    assert isinstance(classify_http_error(error), TransientProcessingError)

def test_connection_failure_is_transient():
    error = httpx.ConnectError("Connection refused")
    assert isinstance(classify_http_error(error), TransientProcessingError)

def test_http_503_is_transient():
    request = httpx.Request("POST", "http://localhost:8000/predict")
    response = httpx.Response(503, request=request)
    error = httpx.HTTPStatusError(
        "Service unavailable", request=request, response=response
    )
    assert isinstance(classify_http_error(error), TransientProcessingError)

def test_http_422_is_permanent():
    request = httpx.Request("POST", "http://localhost:8000/predict")
    response = httpx.Response(422, request=request)
    error = httpx.HTTPStatusError(
        "Invalid features", request=request, response=response
    )
    assert isinstance(classify_http_error(error), PermanentProcessingError)

def test_unexpected_error_is_not_reclassified():
    error = ValueError("Unexpected failure")
    assert classify_http_error(error) is error

def test_retry_succeeds_after_transient_failures():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise TransientProcessingError("Temporary outage")

        return "success"

    with patch("veyra.streaming.retry.time.sleep"):
        result = retry_operation(operation, max_attempts=3)

    assert result == "success"
    assert attempts == 3


def test_retry_stops_after_max_attempts():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise TransientProcessingError("Service unavailable")

    with patch("veyra.streaming.retry.time.sleep"):
        with pytest.raises(TransientProcessingError):
            retry_operation(operation, max_attempts=3)

    assert attempts == 3


def test_permanent_failure_is_not_retried():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise PermanentProcessingError("Invalid payload")

    with patch("veyra.streaming.retry.time.sleep") as mock_sleep:
        with pytest.raises(PermanentProcessingError):
            retry_operation(operation)

    assert attempts == 1
    mock_sleep.assert_not_called()


def test_successful_operation_runs_once():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        return "success"

    assert retry_operation(operation) == "success"
    assert attempts == 1


def test_invalid_max_attempts():
    with pytest.raises(ValueError):
        retry_operation(lambda: "success", max_attempts=0)



def test_scoring_success():
    response = Mock()
    response.json.return_value = {
        "fraud_probability": 0.85,
        "decision": "flagged",
    }

    with patch("veyra.streaming.scoring.httpx.post", return_value=response):
        result = score_application({"income": 50000})

    assert result["fraud_probability"] == 0.85
    assert result["decision"] == "flagged"


def test_scoring_recovers_from_503():
    request = httpx.Request("POST", "http://127.0.0.1:8000/predict")

    unavailable = httpx.Response(503, request=request)
    success = httpx.Response(
        200,
        request=request,
        json={"fraud_probability": 0.25, "decision": "not_flagged"},
    )

    with patch(
        "veyra.streaming.scoring.httpx.post",
        side_effect=[unavailable, success],
    ) as mock_post:
        with patch("veyra.streaming.retry.time.sleep"):
            result = score_application({"income": 50000})

    assert mock_post.call_count == 2
    assert result["fraud_probability"] == 0.25


def test_scoring_rejects_422_without_retry():
    request = httpx.Request("POST", "http://127.0.0.1:8000/predict")
    invalid = httpx.Response(422, request=request)

    with patch(
        "veyra.streaming.scoring.httpx.post",
        return_value=invalid,
    ) as mock_post:
        with pytest.raises(PermanentProcessingError):
            score_application({"income": "invalid"})

    assert mock_post.call_count == 1


def test_dlq_publish_success():
    producer = MagicMock()
    producer.flush.return_value = 0

    event = {"application_id": "123", "features": {"income": 50000}}
    error = ValueError("Invalid features")

    result = publish_to_dlq(
        producer=producer,
        application_id="123",
        original_payload=event,
        error=error,
        attempt_count=1,
    )

    assert result["application_id"] == "123"
    assert result["error_type"] == "ValueError"
    assert result["attempt_count"] == 1
    assert result["original_payload"] == event

    producer.produce.assert_called_once()
    assert producer.produce.call_args.kwargs["topic"] == "veyra.applications.dlq"
    producer.flush.assert_called_once_with(timeout=10.0)


def test_dlq_publish_fails_when_messages_remain():
    producer = MagicMock()
    producer.flush.return_value = 1

    with pytest.raises(RuntimeError, match="DLQ delivery failed"):
        publish_to_dlq(
            producer=producer,
            application_id="123",
            original_payload={"application_id": "123"},
            error=ValueError("Invalid event"),
            attempt_count=1,
        )


def test_dlq_publish_fails_on_delivery_error():
    producer = MagicMock()
    producer.flush.return_value = 0

    def simulate_delivery(**kwargs):
        kwargs["callback"](Exception("Broker unavailable"), None)

    producer.produce.side_effect = simulate_delivery

    with pytest.raises(RuntimeError, match="DLQ delivery failed"):
        publish_to_dlq(
            producer=producer,
            application_id="123",
            original_payload={"application_id": "123"},
            error=ValueError("Invalid event"),
            attempt_count=1,
        )

from veyra.streaming.consumer import publish_prediction


def test_prediction_publish_success():
    producer = MagicMock()
    producer.flush.return_value = 0

    prediction = {
        "fraud_probability": 0.85,
        "decision": "flagged",
    }

    result = publish_prediction(producer, "123", prediction)

    assert result["application_id"] == "123"
    assert result["fraud_probability"] == 0.85

    producer.produce.assert_called_once()
    assert producer.produce.call_args.kwargs["topic"] == "veyra.predictions"
    assert producer.produce.call_args.kwargs["key"] == "123"
    producer.flush.assert_called_once_with(timeout=10.0)


def test_prediction_publish_fails_when_messages_remain():
    producer = MagicMock()
    producer.flush.return_value = 1

    with pytest.raises(RuntimeError, match="Prediction delivery failed"):
        publish_prediction(
            producer,
            "123",
            {"fraud_probability": 0.85},
        )


def test_prediction_publish_fails_on_delivery_error():
    producer = MagicMock()
    producer.flush.return_value = 0

    def simulate_delivery(**kwargs):
        kwargs["callback"](Exception("Broker unavailable"), None)

    producer.produce.side_effect = simulate_delivery

    with pytest.raises(RuntimeError, match="Prediction delivery failed"):
        publish_prediction(
            producer,
            "123",
            {"fraud_probability": 0.85},
        )


def test_recovery_immediate_success():
    with patch(
        "veyra.streaming.recovery.score_application",
        return_value={"fraud_probability": 0.85},
    ) as mock_score:
        result = score_with_recovery({"income": 50000})

    assert result["fraud_probability"] == 0.85
    mock_score.assert_called_once()


def test_recovery_after_transient_failures():
    with patch(
        "veyra.streaming.recovery.score_application",
        side_effect=[
            TransientProcessingError("API unavailable"),
            TransientProcessingError("Timeout"),
            {"fraud_probability": 0.85},
        ],
    ) as mock_score, patch(
        "veyra.streaming.recovery.time.sleep"
    ) as mock_sleep:
        result = score_with_recovery(
            {"income": 50000},
            recovery_delay=15,
        )

    assert result["fraud_probability"] == 0.85
    assert mock_score.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(15)


def test_recovery_does_not_retry_permanent_errors():
    with patch(
        "veyra.streaming.recovery.score_application",
        side_effect=PermanentProcessingError("Invalid features"),
    ) as mock_score, patch(
        "veyra.streaming.recovery.time.sleep"
    ) as mock_sleep:
        with pytest.raises(PermanentProcessingError):
            score_with_recovery({"income": -1})

    mock_score.assert_called_once()
    mock_sleep.assert_not_called()

def test_recovery_exhausted():
    with patch(
        "veyra.streaming.recovery.score_application",
        side_effect=TransientProcessingError("API unavailable"),
    ) as mock_score, patch(
        "veyra.streaming.recovery.time.sleep"
    ) as mock_sleep:

        with pytest.raises(TransientProcessingError):
            score_with_recovery(
                {"income": 50000},
                recovery_delay=15,
                max_recovery_attempts=3,
            )

    assert mock_score.call_count == 3
    assert mock_sleep.call_count == 2