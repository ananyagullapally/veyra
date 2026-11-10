import json
import time
import uuid
from pathlib import Path

from confluent_kafka import Consumer, Producer, KafkaException
from veyra.streaming import consumer as scoring

BOOTSTRAP = "127.0.0.1:9092"
INPUT = "veyra.benchmark.applications"
OUTPUT = "veyra.benchmark.predictions"
COUNT = 100

payload = json.loads(
    Path("/tmp/veyra_benchmark_payload.json").read_text()
)
features = payload["features"]

run_id = uuid.uuid4().hex[:12]

producer = Producer({
    "bootstrap.servers": BOOTSTRAP,
    "acks": "all",
    "enable.idempotence": True,
})

consumer = Consumer({
    "bootstrap.servers": BOOTSTRAP,
    "group.id": f"veyra-benchmark-{run_id}",
    "auto.offset.reset": "latest",
    "enable.auto.commit": False,
})

# Temporarily redirect the existing prediction publisher.
scoring.PREDICTION_TOPIC = OUTPUT

consumer.subscribe([INPUT])

# Establish partition assignment before publishing benchmark events.
deadline = time.monotonic() + 15
while not consumer.assignment():
    consumer.poll(0.5)
    if time.monotonic() > deadline:
        consumer.close()
        raise RuntimeError("Benchmark consumer assignment timed out")

try:
    print(f"Publishing {COUNT} benchmark applications...")

    for i in range(COUNT):
        application_id = f"benchmark-{run_id}-{i}"

        event = {
            "application_id": application_id,
            "features": features,
            "source": "benchmark",
        }

        producer.produce(
            INPUT,
            key=application_id,
            value=json.dumps(event).encode(),
        )
        producer.poll(0)

    if producer.flush(30) != 0:
        raise RuntimeError("Benchmark input delivery timed out")

    completed = 0
    start = time.perf_counter()
    deadline = time.monotonic() + 120

    while completed < COUNT:
        if time.monotonic() > deadline:
            raise TimeoutError("Benchmark processing timed out")

        message = consumer.poll(1.0)

        if message is None:
            continue

        if message.error():
            raise KafkaException(message.error())

        result = scoring.process_application_message(
            message, consumer, producer
        )

        if result != "predicted":
            raise RuntimeError(f"Unexpected result: {result}")

        completed += 1

    elapsed = time.perf_counter() - start

    print("\nVEYRA KAFKA THROUGHPUT BENCHMARK")
    print("--------------------------------")
    print(f"Completed events: {completed}")
    print(f"Elapsed time:     {elapsed:.3f} seconds")
    print(f"Throughput:       {completed / elapsed:.2f} events/sec")
    print("Consumer count:   1")
    print("Concurrency:      Sequential")
    print("Failures:         0")

finally:
    consumer.close()
