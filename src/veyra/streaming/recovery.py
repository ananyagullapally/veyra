import time

from veyra.streaming.errors import TransientProcessingError
from veyra.streaming.scoring import score_application


def score_with_recovery(
    features,
    recovery_delay=15,
    max_recovery_attempts=3,
):
    if max_recovery_attempts < 1:
        raise ValueError("max_recovery_attempts must be at least 1")

    for attempt in range(1, max_recovery_attempts + 1):
        try:
            return score_application(features)

        except TransientProcessingError as error:
            if attempt == max_recovery_attempts:
                raise

            print(f"API temporarily unavailable: {error}")
            print(f"Retrying after {recovery_delay} seconds...")
            time.sleep(recovery_delay)