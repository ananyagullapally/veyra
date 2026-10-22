from contextlib import contextmanager

from veyra.storage.database import get_connection


RETRAINING_LOCK_ID = 8675309


@contextmanager
def acquire_retraining_lock():
    """
    Prevent overlapping retraining jobs using
    a PostgreSQL session-level advisory lock.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT pg_try_advisory_lock(%s)",
                (RETRAINING_LOCK_ID,),
            )
            acquired = cursor.fetchone()[0]

        if not acquired:
            raise RuntimeError(
                "Another Veyra retraining job is already running."
            )

        try:
            yield
        finally:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_unlock(%s)",
                    (RETRAINING_LOCK_ID,),
                )