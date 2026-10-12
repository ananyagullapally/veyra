import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from veyra.training.mlflow_setup import configure_mlflow


MODEL_NAME = "veyra-fraud-detector"
MODEL_ALIAS = "champion"


def load_champion():

    configure_mlflow()

    client = MlflowClient()

    # Find the registered champion version
    version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    # Load the complete preprocessing + XGBoost pipeline
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    pipeline = mlflow.sklearn.load_model(model_uri)

    # Retrieve the threshold from the training run
    run = client.get_run(version.run_id)

    threshold = float(
        run.data.params["selected_threshold"]
    )

    print(f"Loaded champion model version {version.version}")
    print(f"Decision threshold: {threshold:.6f}")

    return pipeline, threshold
