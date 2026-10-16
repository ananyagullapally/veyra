from veyra.monitoring import config
from veyra.storage.database import get_connection


def get_prediction_metrics() -> dict:
    """Calculate metrics for recent production predictions."""

    query = """
        SELECT
            COUNT(*) AS total_predictions,
            COUNT(*) FILTER (
                WHERE decision = 'flagged'
            ) AS flagged,
            COUNT(*) FILTER (
                WHERE decision = 'not_flagged'
            ) AS not_flagged,
            AVG(fraud_probability) AS avg_fraud_probability
        FROM fraud_predictions
        WHERE source = 'production'
          AND predicted_at >= NOW() - (%s * INTERVAL '1 hour');
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (config.MONITORING_WINDOW_HOURS,))
            total, flagged, not_flagged, avg_probability = cursor.fetchone()

    return {
        "total_predictions": total,
        "flagged": flagged,
        "not_flagged": not_flagged,
        "flagged_rate": flagged / total if total else 0.0,
        "avg_fraud_probability": (
            float(avg_probability) if avg_probability is not None else 0.0
        ),
    }