import random
import time

from veyra.streaming.errors import TransientProcessingError


def retry_operation(
    operation,
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()

        except TransientProcessingError:
            if attempt == max_attempts:
                raise

            delay = min(
                base_delay * (2 ** (attempt - 1)),
                max_delay,
            )

            jitter = random.uniform(0, delay * 0.2)
            time.sleep(delay + jitter)