import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold_for_recall(
    y_true,
    probabilities,
    minimum_recall: float = 0.80,
) -> dict:
    """Select the highest-precision threshold meeting minimum recall."""

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError(
            "minimum_recall must be between 0 and 1."
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    # precision_recall_curve returns one extra precision/recall
    # value compared with the threshold array.
    precision = precision[:-1]
    recall = recall[:-1]

    valid_indices = np.where(recall >= minimum_recall)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No threshold achieves recall >= {minimum_recall:.2f}"
        )

    best_index = valid_indices[
        np.argmax(precision[valid_indices])
    ]

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
    }