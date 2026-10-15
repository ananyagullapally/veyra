
import json
import logging
import os

from kafka import KafkaConsumer

from veyra.storage.predictions import save_prediction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_storage_consumer():
    consumer = KafkaConsumer(
        "veyra.predictions",
        bootstrap_servers=os.getenv(
            "VEYRA_KAFKA_BOOTSTRAP_SERVERS",
            "127.0.0.1:9092",
        ),
        group_id="veyra-prediction-storage",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )

    logger.info("Prediction storage consumer started")

    try:
        for message in consumer:
            try:
                prediction = json.loads(message.value.decode("utf-8"))
                save_prediction(prediction)

                # Commit only after PostgreSQL successfully commits.
                consumer.commit()

                logger.info(
                    "Stored prediction for application %s",
                    prediction["application_id"],
                )

            except Exception:
                logger.exception(
                    "Failed to store prediction; offset not committed"
                )
                raise

    finally:
        consumer.close()


if __name__ == "__main__":
    run_storage_consumer()
