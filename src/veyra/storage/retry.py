
import logging
import time

import psycopg

from veyra.storage.predictions import save_prediction


logger = logging.getLogger(__name__)


def save_prediction_with_retry(
    prediction: dict,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> None:
    """Retry transient PostgreSQL errors with exponential backoff."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            save_prediction(prediction)
            return

        except psycopg.OperationalError:
            if attempt == max_attempts:
                logger.exception(
                    "PostgreSQL unavailable after %s attempts",
                    max_attempts,
                )
                raise

            delay = base_delay * (2 ** (attempt - 1))

            logger.warning(
                "PostgreSQL connection failed; retrying in %.1f seconds "
                "(attempt %s/%s)",
                delay,
                attempt,
                max_attempts,
            )

            time.sleep(delay)
