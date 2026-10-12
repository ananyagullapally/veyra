import mlflow


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "veyra-fraud-detection"


def configure_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)