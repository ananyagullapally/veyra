
import os
from unittest.mock import patch

import pytest
from veyra.monitoring.drift_monitor import check_prediction_drift
from veyra.storage.database import get_connection
from veyra.monitoring.metrics import get_prediction_metrics


@pytest.mark.integration
def test_production_filter_and_monitoring_window():
    if os.getenv("VEYRA_RUN_DB_TESTS") != "1":
        pytest.skip("Set VEYRA_RUN_DB_TESTS=1 to run database tests")

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE fraud_predictions (
                    application_id TEXT,
                    source TEXT,
                    fraud_probability DOUBLE PRECISION,
                    decision TEXT,
                    predicted_at TIMESTAMPTZ
                ) ON COMMIT PRESERVE ROWS;
            """)

            cur.execute("""
                INSERT INTO fraud_predictions
                    (application_id, source, fraud_probability,
                     decision, predicted_at)
                VALUES
                    ('recent-prod', 'production', 0.80,
                     'flagged', NOW()),
                    ('old-prod', 'production', 0.70,
                     'flagged', NOW() - INTERVAL '48 hours'),
                    ('recent-pipeline', 'pipeline_test', 0.60,
                     'not_flagged', NOW()),
                    ('recent-synthetic', 'synthetic_test', 0.40,
                     'not_flagged', NOW());
            """)

        # Keep the temporary table visible after commit.
        conn.commit()

        with patch(
            "veyra.monitoring.metrics.get_connection",
            return_value=conn,
        ):
            metrics = get_prediction_metrics()

        assert metrics["total_predictions"] == 1
        assert metrics["flagged"] == 1
        assert metrics["not_flagged"] == 0
        assert metrics["flagged_rate"] == pytest.approx(1.0)
        assert metrics["avg_fraud_probability"] == pytest.approx(0.80)

    finally:
        conn.close()


@pytest.mark.integration
def test_drift_monitor_production_filter():
    if os.getenv("VEYRA_RUN_DB_TESTS") != "1":
        pytest.skip("Set VEYRA_RUN_DB_TESTS=1 to run database tests")

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE fraud_predictions (
                    application_id TEXT,
                    source TEXT,
                    fraud_probability DOUBLE PRECISION,
                    predicted_at TIMESTAMPTZ
                ) ON COMMIT PRESERVE ROWS;
            """)

            cur.execute("""
                INSERT INTO fraud_predictions
                    (application_id, source, fraud_probability, predicted_at)
                VALUES
                    ('recent-prod', 'production', 0.80, NOW()),
                    ('old-prod', 'production', 0.70,
                     NOW() - INTERVAL '48 hours'),
                    ('recent-pipeline', 'pipeline_test', 0.60, NOW()),
                    ('recent-synthetic', 'synthetic_test', 0.40, NOW());
            """)

        conn.commit()

        with patch(
            "veyra.monitoring.drift_monitor.get_connection",
            return_value=conn,
        ):
            result = check_prediction_drift()

        assert result["status"] == "insufficient_data"
        assert result["sample_count"] == 1
        assert result["minimum_required"] == 100
        assert result["psi"] is None

    finally:
        conn.close()

