import json
import time
from confluent_kafka import Consumer, Producer, KafkaException
from veyra.streaming.recovery import score_with_recovery
from veyra.streaming.dlq import publish_to_dlq
from veyra.streaming.errors import PermanentProcessingError
from veyra.streaming.errors import TransientProcessingError

KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
APPLICATION_TOPIC = "veyra.applications"
PREDICTION_TOPIC = "veyra.predictions"


def parse_application_event(raw_value: bytes) -> dict | None:
    try:
        event = json.loads(raw_value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(event, dict):
        return None

    if "features" not in event or not isinstance(event["features"], dict):
        return None

    return event


def publish_prediction(producer, application_id, prediction, source="unknown"):
    prediction_event = {
        "application_id": application_id,
        **prediction,
        "source": source,
    }

    delivery_errors = []

    def delivery_report(error, delivered_message):
        if error is not None:
            delivery_errors.append(str(error))

    producer.produce(
        topic=PREDICTION_TOPIC,
        key=str(application_id),
        value=json.dumps(prediction_event).encode("utf-8"),
        callback=delivery_report,
    )

    remaining = producer.flush(timeout=10.0)

    if remaining > 0 or delivery_errors:
        raise RuntimeError(
            f"Prediction delivery failed: {delivery_errors}"
        )

    return prediction_event


def process_application_message(message, consumer, producer):
    event = parse_application_event(message.value())

    if event is None:
        raw_payload = message.value()
        application_id = (
            message.key().decode("utf-8", errors="replace")
            if message.key() is not None
            else f"{message.topic()}-{message.partition()}-{message.offset()}"
        )

        publish_to_dlq(
            producer=producer,
            application_id=application_id,
            original_payload={
                "raw_message": raw_payload.decode("utf-8", errors="replace")
            },
            error=ValueError("Invalid application event"),
            attempt_count=1,
        )

        consumer.commit(message=message, asynchronous=False)
        return "dlq"

    application_id = event.get("application_id")

    try:
        prediction = score_with_recovery(event["features"])

    except PermanentProcessingError as error:
        publish_to_dlq(
            producer=producer,
            application_id=application_id,
            original_payload=event,
            error=error,
            attempt_count=1,
        )

        consumer.commit(message=message, asynchronous=False)
        return "dlq"

    publish_prediction(
        producer=producer,
        application_id=application_id,
        prediction=prediction,
        source=event.get("source", "unknown"),
    )

    consumer.commit(message=message, asynchronous=False)
    return "predicted"


def main():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "veyra-fraud-scoring",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    prediction_producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "acks": "all",
        "enable.idempotence": True,
    })

    consumer.subscribe([APPLICATION_TOPIC])
    print("Veyra consumer started. Waiting for applications...")

    try:
        while True:
            message = consumer.poll(timeout=1.0)

            if message is None:
                continue

            if message.error():
                raise KafkaException(message.error())

            result = process_application_message(
                message=message,
                consumer=consumer,
                producer=prediction_producer,
            )

            print(f"Application processing result: {result}")

    except KeyboardInterrupt:
        print("\nStopping Veyra consumer...")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
