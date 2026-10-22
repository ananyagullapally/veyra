from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    pipeline,
    X,
    y,
    threshold: float,
) -> dict:
    """Evaluate a model using a fixed classification threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    probabilities = pipeline.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "threshold": float(threshold),
    }