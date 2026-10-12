from mlflow import MlflowClient
from veyra.training.mlflow_setup import configure_mlflow


def main():
    configure_mlflow()

    client = MlflowClient()

    client.set_registered_model_alias(
        name="veyra-fraud-detector",
        alias="champion",
        version="1",
    )

    print("Champion alias assigned to model version 1!")


if __name__ == "__main__":
    main()