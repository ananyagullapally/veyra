import json
from unittest.mock import MagicMock

from veyra.streaming.consumer import publish_prediction


def test_prediction_preserves_source():
    producer = MagicMock()
    producer.flush.return_value = 0

    prediction = {
        "fraud_probability": 0.85,
        "decision": "flagged",
        "source": "production",
    }

    result = publish_prediction(
        producer=producer,
        application_id="source-test-001",
        prediction=prediction,
        source="pipeline_test",
    )

    assert result["source"] == "pipeline_test"
    assert result["application_id"] == "source-test-001"

    published = producer.produce.call_args.kwargs
    payload = json.loads(published["value"].decode("utf-8"))

    assert payload["source"] == "pipeline_test"
    assert payload["fraud_probability"] == 0.85