import mlflow

from veyra.training.mlflow_setup import configure_mlflow


def main():
    # Connect to our local MLflow tracking server
    configure_mlflow()

    # Create a test experiment run
    with mlflow.start_run(run_name="connection-test"):

        # Log sample model parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("max_depth", 3)

        # Log a dummy metric to verify tracking
        mlflow.log_metric("test_metric", 0.95)

    print("MLflow test run logged successfully!")


if __name__ == "__main__":
    main()