from mlflow import MlflowClient
import math
from veyra.training.mlflow_setup import configure_mlflow


def get_candidate_info(run_id: str) -> dict:
    configure_mlflow()
    client = MlflowClient()

    run = client.get_run(run_id)

    required_metrics = (
        "validation_pr_auc",
        "validation_precision",
        "validation_recall",
    )

    for metric in required_metrics:
        if metric not in run.data.metrics:
            raise ValueError(f"Candidate is missing metric: {metric}")

    threshold = run.data.params.get("selected_threshold")

    if threshold is None:
        raise ValueError("Candidate is missing selected_threshold")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("Candidate threshold must be numeric.")

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Candidate threshold must be finite and between 0 and 1.")

    return {
        "run_id": run_id,
        "metrics": {
            "pr_auc": run.data.metrics["validation_pr_auc"],
            "precision": run.data.metrics["validation_precision"],
            "recall": run.data.metrics["validation_recall"],
        },
        "threshold": float(threshold),
    }