import numpy as np
import mlflow
import mlflow.sklearn

from veyra.training.mlflow_setup import configure_mlflow
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)


def main():
    configure_mlflow()

    # Find the latest successfully logged model
    experiment = mlflow.get_experiment_by_name(
        "veyra-fraud-detection"
    )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'xgboost-depth3-final'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        raise ValueError("No XGBoost MLflow run found.")

    # Find a run that actually has a logged model
    model_uri = None

    for run_id in runs["run_id"]:
        logged_models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
        )

        if not logged_models.empty:
            model_id = logged_models.iloc[0]["model_id"]
            model_uri = f"models:/{model_id}"
            break

    if model_uri is None:
        raise ValueError("No logged fraud model found.")

    print(f"Loading model: {model_uri}")

    pipeline = mlflow.sklearn.load_model(
    model_uri,
    )

    # Small inference sample from existing validation data
    validation_df = load_split_data("validation")
    X_validation, _ = create_feature_target_split(
        validation_df
    )

    sample = X_validation.head(5)

    scores = pipeline.predict_proba(sample)[:, 1]

    print("\nFraud scores:")
    print(scores)

    assert len(scores) == 5
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))

    print("\nMLflow model loaded and inference successful!")


if __name__ == "__main__":
    main()