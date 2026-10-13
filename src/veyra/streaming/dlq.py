import json
from datetime import datetime, timezone


DLQ_TOPIC = "veyra.applications.dlq"


def publish_to_dlq(
    producer,
    application_id,
    original_payload,
    error,
    attempt_count,
):
    dlq_event = {
        "application_id": application_id,
        "original_payload": original_payload,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "attempt_count": attempt_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    delivery_errors = []

    def delivery_report(err, message):
        if err is not None:
            delivery_errors.append(str(err))

    producer.produce(
        topic=DLQ_TOPIC,
        key=str(application_id),
        value=json.dumps(dlq_event).encode("utf-8"),
        callback=delivery_report,
    )

    remaining = producer.flush(timeout=10.0)

    if remaining > 0 or delivery_errors:
        raise RuntimeError(
            f"DLQ delivery failed: {delivery_errors}"
        )

    return dlq_event