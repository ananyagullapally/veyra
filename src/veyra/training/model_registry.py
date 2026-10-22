from mlflow import MlflowClient

from veyra.training import threshold
from veyra.training.mlflow_setup import configure_mlflow


MODEL_NAME = "veyra-fraud-detector"


def get_champion_info() -> dict:
    """Return information about the current champion model."""

    configure_mlflow()

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion",
    )

    run = client.get_run(champion.run_id)

    threshold_value = run.data.params.get("selected_threshold")

    if threshold_value is None:
        raise ValueError(
            "Champion model is missing selected_threshold in MLflow."
        )

    threshold = float(threshold_value)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Champion threshold must be between 0 and 1.")

    return {
        "model_name": champion.name,
        "version": champion.version,
        "run_id": champion.run_id,
        "source": champion.source,
        "model_uri": f"models:/{MODEL_NAME}@champion",
        "threshold": threshold,
    }