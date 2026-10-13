import json
from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
APPLICATION_TOPIC = "veyra.applications"


def delivery_report(error, message):
    if error:
        print(f"Delivery failed: {error}")
    else:
        print(
            f"Delivered to {message.topic()} "
            f"partition {message.partition()} "
            f"offset {message.offset()}"
        )


def main():
    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    })

    event = {
        "application_id": "demo-002",
        "event_type": "application_submitted",
        "source": "synthetic",
    }

    producer.produce(
        topic=APPLICATION_TOPIC,
        key=event["application_id"],
        value=json.dumps(event).encode("utf-8"),
        callback=delivery_report,
    )

    producer.flush()


if __name__ == "__main__":
    main()