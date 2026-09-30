import numpy as np
import pytest
from veyra.training.threshold import select_threshold_for_recall


def test_select_threshold_meets_minimum_recall():
    y_true = np.array([0, 0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.20,
        0.40,
        0.70,
        0.90,
    ])

    result = select_threshold_for_recall(
        y_true,
        probabilities,
        minimum_recall=0.80,
    )

    assert result["recall"] >= 0.80
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["threshold"] <= 1.0

def test_invalid_recall_target_raises_error():
    y_true = np.array([0, 1])
    probabilities = np.array([0.20, 0.80])

    with pytest.raises(ValueError):
        select_threshold_for_recall(
            y_true,
            probabilities,
            minimum_recall=1.10,
        )