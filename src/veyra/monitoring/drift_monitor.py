from pathlib import Path

import pandas as pd

from veyra.monitoring.drift import calculate_psi
from veyra.storage.database import get_connection
from veyra.monitoring import config

REFERENCE_PATH = Path("data/processed/drift_reference.parquet")

def check_prediction_drift() -> dict:
    """Compare stored prediction scores against validation reference."""

    reference_df = pd.read_parquet(REFERENCE_PATH)
    reference = reference_df["fraud_probability"].to_numpy()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
            """
            SELECT fraud_probability
            FROM fraud_predictions
            WHERE source = 'production'
                AND predicted_at >= NOW() - (%s * INTERVAL '1 hour')
            ORDER BY predicted_at DESC;
            """,
            (config.MONITORING_WINDOW_HOURS,),
            )
            rows = cursor.fetchall()

    current = [row[0] for row in rows]

    if len(current) < config.MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "sample_count": len(current),
            "minimum_required": config.MIN_SAMPLES,
            "psi": None,
        }

    psi = calculate_psi(reference, current)

    if psi < 0.10:
        status = "stable"
    elif psi < 0.25:
        status = "moderate_drift"
    else:
        status = "significant_drift"

    return {
        "status": status,
        "sample_count": len(current),
        "minimum_required": config.MIN_SAMPLES,
        "psi": psi,
    }