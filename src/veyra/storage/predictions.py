
from veyra.storage.database import get_connection


def save_prediction(prediction: dict) -> None:
    """Insert or update a fraud prediction by application ID."""

    query = """
        INSERT INTO fraud_predictions (
            application_id,
            fraud_probability,
            decision,
            threshold,
            model,
            alias
        )
        VALUES (
            %(application_id)s,
            %(fraud_probability)s,
            %(decision)s,
            %(threshold)s,
            %(model)s,
            %(alias)s
        )
        ON CONFLICT (application_id)
        DO UPDATE SET
            fraud_probability = EXCLUDED.fraud_probability,
            decision = EXCLUDED.decision,
            threshold = EXCLUDED.threshold,
            model = EXCLUDED.model,
            alias = EXCLUDED.alias,
            updated_at = NOW();
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, prediction)
