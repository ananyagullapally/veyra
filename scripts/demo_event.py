
import json
import uuid
from pathlib import Path

from confluent_kafka import Producer

BOOTSTRAP = "127.0.0.1:9092"
INPUT_TOPIC = "veyra.applications"

payload_path = Path("/tmp/veyra_benchmark_payload.json")

if not payload_path.exists():
    raise FileNotFoundError(
        "Demo payload missing. Recreate the benchmark payload first."
    )

payload = json.loads(payload_path.read_text())

application_id = f"demo-{uuid.uuid4().hex[:8]}"

event = {
    "application_id": application_id,
    "features": payload["features"],
    "source": "portfolio-demo",
}

producer = Producer({"bootstrap.servers": BOOTSTRAP})

delivery_errors = []

def delivery_callback(error, message):
    if error:
        delivery_errors.append(str(error))

producer.produce(
    topic=INPUT_TOPIC,
    key=application_id,
    value=json.dumps(event).encode("utf-8"),
    callback=delivery_callback,
)

remaining = producer.flush(timeout=10)

if remaining or delivery_errors:
    raise RuntimeError(
        f"Kafka delivery failed: {delivery_errors}"
    )

print("\nVEYRA | LIVE FRAUD DETECTION")
print("----------------------------")
print(f"Application ID: {application_id}")
print(f"Kafka topic:    {INPUT_TOPIC}")
print("Status:         Published successfully")
print("Consumer:       Waiting for prediction")
