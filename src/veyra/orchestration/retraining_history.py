from datetime import datetime

from veyra.storage.database import get_connection


def get_last_retraining_at() -> datetime | None:
    """
    Return the start time of the latest retraining attempt.

    Includes RUNNING, COMPLETED, and FAILED attempts
    so failed jobs also count toward the cooldown.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT started_at
                FROM retraining_history
                ORDER BY started_at DESC, id DESC
                LIMIT 1
                """
            )

            row = cursor.fetchone()

    return row[0] if row else None

def start_retraining_attempt() -> int:
    """
    Record a new retraining attempt and return its database ID.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO retraining_history (status)
                VALUES ('RUNNING')
                RETURNING id
                """
            )
            attempt_id = cursor.fetchone()[0]

    return attempt_id

def complete_retraining_attempt(
    attempt_id: int,
    candidate_run_id: str,
    action: str,
) -> None:
    """Mark a retraining attempt as successfully completed."""

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE retraining_history
                SET status = 'COMPLETED',
                    finished_at = NOW(),
                    candidate_run_id = %s,
                    action = %s
                WHERE id = %s
                  AND status = 'RUNNING'
                """,
                (candidate_run_id, action, attempt_id),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    f"No RUNNING retraining attempt found for ID {attempt_id}."
                )
def fail_retraining_attempt(
    attempt_id: int,
    error_message: str,
) -> None:
    """Mark a retraining attempt as failed."""

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE retraining_history
                SET status = 'FAILED',
                    finished_at = NOW(),
                    error_message = %s
                WHERE id = %s
                  AND status = 'RUNNING'
                """,
                (error_message, attempt_id),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    f"No RUNNING retraining attempt found for ID {attempt_id}."
                )