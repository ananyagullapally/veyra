import json

from veyra.streaming.consumer import parse_application_event


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