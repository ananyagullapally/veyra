
import json
import httpx
import pandas as pd
from confluent_kafka import Producer


def main():
    applications = pd.read_parquet(
        "data/processed/applications.parquet"
    )

    # Use one validation application for a local replay test
    sample = applications[
    applications["dataset_split"] == "validation"
    ].iloc[1]

    event = {
        "application_id": str(sample["application_id"]),
        "features": {},
    }

    excluded = {
    "application_id",
    "dataset_split",
    "fraud_bool",
    "month",
    }

    for column in sample.index:
        if column not in excluded:
            value = sample[column]
            event["features"][column] = (
                None if pd.isna(value)
                else value.item() if hasattr(value, "item")
                else value
            )

    producer = Producer({
        "bootstrap.servers": "127.0.0.1:9092"
    })

    producer.produce(
        "veyra.applications",
        key=event["application_id"],
        value=json.dumps(event).encode("utf-8"),
    )

    producer.flush()

    print("\nReceived application event:")
    print("Application ID:", event.get("application_id"))
    print("Event type:", event.get("event_type"))
    print("Source:", event.get("source"))



    if "features" in event:
        response = httpx.post(
        "http://127.0.0.1:8000/predict",
        json={"features": event["features"]},
        timeout=10.0,
        )

        response.raise_for_status()
        prediction = response.json()

        print("Fraud prediction:", prediction)
    else:
        print("No features provided. Skipping scoring.")

if __name__ == "__main__":
    main()
