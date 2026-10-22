import mlflow

from veyra.training.mlflow_setup import configure_mlflow
from veyra.training.model_registry import get_champion_info
from veyra.training.candidate_registry import get_candidate_info
from veyra.training.evaluate_model import evaluate_model
from veyra.training.compare_models import compare_models
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)
from mlflow import MlflowClient

def evaluate_candidate(run_id: str) -> dict:
    configure_mlflow()

    champion = get_champion_info()
    candidate = get_candidate_info(run_id)

    client = MlflowClient()

    logged_models = client.search_logged_models(
        experiment_ids=["1"],
        filter_string=f"source_run_id = '{run_id}'",
    )

    ready_models = [
        model for model in logged_models
        if model.status == "READY"
    ]

    if len(ready_models) != 1:
        raise ValueError(
            "Candidate must have exactly one READY logged model."
        )

    candidate_model = mlflow.sklearn.load_model(
        f"models:/{ready_models[0].model_id}"
    )

    if not callable(getattr(candidate_model, "predict_proba", None)):
        raise ValueError(
            "Candidate model does not support predict_proba."
        )

    validation_df = load_split_data("validation")
    X_val, y_val = create_feature_target_split(validation_df)

    champion_model = mlflow.sklearn.load_model(
        champion["model_uri"]
    )

    champion_metrics = evaluate_model(
        champion_model,
        X_val,
        y_val,
        champion["threshold"],
    )

    candidate_metrics = evaluate_model(
    candidate_model,
    X_val,
    y_val,
    candidate["threshold"],
)

    comparison = compare_models(
        champion_metrics=champion_metrics,
        candidate_metrics=candidate_metrics,
    )

    return {
        "champion_version": champion["version"],
        "candidate_run_id": candidate["run_id"],
        "comparison": comparison,
    }