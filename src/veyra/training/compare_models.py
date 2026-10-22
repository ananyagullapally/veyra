def compare_models(
    champion_metrics: dict,
    candidate_metrics: dict,
    minimum_recall: float = 0.80,
) -> dict:
    """Compare candidate performance against the champion."""

    required_metrics = ("pr_auc", "precision", "recall")

    for model_name, metrics in (
        ("champion", champion_metrics),
        ("candidate", candidate_metrics),
    ):
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(
                    f"{model_name} is missing metric: {metric}"
                )

    champion_pr_auc = champion_metrics["pr_auc"]
    candidate_pr_auc = candidate_metrics["pr_auc"]
    candidate_recall = candidate_metrics["recall"]

    passes_recall = candidate_recall >= minimum_recall
    improves_pr_auc = candidate_pr_auc > champion_pr_auc

    eligible = passes_recall and improves_pr_auc

    return {
        "champion_pr_auc": champion_pr_auc,
        "candidate_pr_auc": candidate_pr_auc,
        "pr_auc_improvement": (
            candidate_pr_auc - champion_pr_auc
        ),
        "candidate_precision": candidate_metrics["precision"],
        "candidate_recall": candidate_recall,
        "passes_recall": passes_recall,
        "improves_pr_auc": improves_pr_auc,
        "eligible_for_promotion": eligible,
    }