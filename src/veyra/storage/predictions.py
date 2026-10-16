from veyra.storage.database import get_connection


VALID_SOURCES = {
    "unknown",
    "production",
    "pipeline_test",
    "synthetic_test",
}


def save_prediction(prediction: dict) -> None:
    """Store a prediction without overwriting an existing application."""

    source = prediction.get("source", "unknown")

    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid prediction source: {source}")

    query = """
        INSERT INTO fraud_predictions (
            application_id,
            fraud_probability,
            decision,
            threshold,
            model,
            alias,
            source
        )
        VALUES (
            %(application_id)s,
            %(fraud_probability)s,
            %(decision)s,
            %(threshold)s,
            %(model)s,
            %(alias)s,
            %(source)s
        )
        ON CONFLICT (application_id)
        DO NOTHING;
    """

    parameters = {
        **prediction,
        "source": source,
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)