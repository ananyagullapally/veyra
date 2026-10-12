import mlflow
from veyra.training.mlflow_setup import configure_mlflow


def main():
    configure_mlflow()

    model_uri = "models:/m-90ef40d968b04683a42d626a18c79c05"
    model_name = "veyra-fraud-detector"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(f"Registered model: {result.name}")
    print(f"Version: {result.version}")


if __name__ == "__main__":
    main()