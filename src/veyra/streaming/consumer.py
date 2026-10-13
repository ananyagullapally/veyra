
import json

import httpx
from confluent_kafka import Consumer, Producer, KafkaException


KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
APPLICATION_TOPIC = "veyra.applications"
PREDICTION_TOPIC = "veyra.predictions"
API_URL = "http://127.0.0.1:8000/predict"


def main():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "veyra-fraud-scoring",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    prediction_producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
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

            event = parse_application_event(message.value())

            if event is None:
                print("Invalid application event. Skipping.")
                consumer.commit(message=message, asynchronous=False)
                continue

            application_id = event.get("application_id")
            print("\nReceived application:", application_id)

            # 1. Score the application using FastAPI
            response = httpx.post(
                API_URL,
                json={"features": event["features"]},
                timeout=10.0,
            )
            response.raise_for_status()

            prediction = response.json()
            print("Fraud prediction:", prediction)

            # 2. Build the prediction event
            prediction_event = {
                "application_id": application_id,
                **prediction,
            }

            # 3. Publish the prediction to Kafka
            delivery_errors = []

            def delivery_report(error, delivered_message):
                if error is not None:
                    delivery_errors.append(str(error))

            prediction_producer.produce(
                topic=PREDICTION_TOPIC,
                key=str(application_id),
                value=json.dumps(prediction_event).encode("utf-8"),
                callback=delivery_report,
            )

            remaining = prediction_producer.flush(timeout=10.0)

            if remaining > 0 or delivery_errors:
                raise RuntimeError(
                    f"Prediction delivery failed: {delivery_errors}"
                )

            print("Prediction published to Kafka")

            # 4. Commit only after successful publication
            consumer.commit(
                message=message,
                asynchronous=False,
            )

    except KeyboardInterrupt:
        print("\nStopping Veyra consumer...")

    finally:
        consumer.close()

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


if __name__ == "__main__":
    main()
