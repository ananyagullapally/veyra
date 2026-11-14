import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from veyra.serving.model_loader import load_champion
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)

# Load the registered champion and its original threshold
pipeline, threshold = load_champion()

# Load the original month-6 validation split
validation_df = load_split_data("validation")
X_validation, y_validation = create_feature_target_split(validation_df)

# Evaluate the existing model without fitting or retraining
probabilities = pipeline.predict_proba(X_validation)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "validation_pr_auc": float(
        average_precision_score(y_validation, probabilities)
    ),
    "validation_roc_auc": float(
        roc_auc_score(y_validation, probabilities)
    ),
    "validation_precision": float(
        precision_score(y_validation, predictions, zero_division=0)
    ),
    "validation_recall": float(
        recall_score(y_validation, predictions, zero_division=0)
    ),
    "validation_f1": float(
        f1_score(y_validation, predictions, zero_division=0)
    ),
}

client = MlflowClient()
version = client.get_model_version_by_alias(
    name="veyra-fraud-detector",
    alias="champion",
)

print("\nChampion validation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print(f"\nChampion version: {version.version}")
print(f"Source run ID: {version.run_id}")
print(f"Threshold: {threshold:.8f}")

# Safety: do not log to an unexpected run
EXPECTED_RUN_ID = "a7ab4a7999ae4a23830a450eef4b1d14"

if version.run_id != EXPECTED_RUN_ID:
    raise RuntimeError(
        "Champion source run changed. Refusing to log metrics."
    )

# Add evaluation metrics to the existing champion run
for name, value in metrics.items():
    client.log_metric(version.run_id, name, value)

print("\nValidation metrics logged to the existing champion run.")
print("Model artifacts and champion alias remain unchanged.")
