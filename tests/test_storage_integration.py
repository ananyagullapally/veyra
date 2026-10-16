
import os
import uuid

import pytest

from veyra.storage.database import get_connection
from veyra.storage.predictions import save_prediction


@pytest.mark.integration
def test_prediction_insert_and_replay():
    if os.getenv("VEYRA_RUN_DB_TESTS") != "1":
        pytest.skip("Set VEYRA_RUN_DB_TESTS=1 to run database tests")

    application_id = f"integration-{uuid.uuid4()}"

    prediction = {
        "application_id": application_id,
        "fraud_probability": 0.85,
        "decision": "flagged",
        "threshold": 0.01320815272629261,
        "model": "veyra-fraud-detector",
        "alias": "champion",
    }

    try:
        save_prediction(prediction)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fraud_probability, updated_at
                    FROM fraud_predictions
                    WHERE application_id = %s
                    """,
                    (application_id,),
                )
                before = cur.fetchone()

        assert before is not None
        assert before[0] == pytest.approx(0.85)

        prediction["fraud_probability"] = 0.10
        save_prediction(prediction)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fraud_probability, updated_at
                    FROM fraud_predictions
                    WHERE application_id = %s
                    """,
                    (application_id,),
                )
                after = cur.fetchone()

        assert after == before

    finally:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM fraud_predictions WHERE application_id = %s",
                    (application_id,),
                )
