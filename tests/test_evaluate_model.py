import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from unittest.mock import Mock

from veyra.training.evaluate_model import evaluate_model


def test_evaluate_model():
    probabilities = np.array([0.10, 0.80, 0.30, 0.90])
    y_true = np.array([0, 1, 0, 1])

    pipeline = Mock()
    pipeline.predict_proba.return_value = np.column_stack(
        [1 - probabilities, probabilities]
    )

    X = np.zeros((4, 2))

    result = evaluate_model(
        pipeline=pipeline,
        X=X,
        y=y_true,
        threshold=0.50,
    )

    assert result["pr_auc"] == pytest.approx(
        average_precision_score(y_true, probabilities)
    )
    assert result["roc_auc"] == pytest.approx(
        roc_auc_score(y_true, probabilities)
    )
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["threshold"] == pytest.approx(0.50)


def test_evaluate_model_invalid_threshold():
    pipeline = Mock()

    with pytest.raises(ValueError, match="threshold"):
        evaluate_model(
            pipeline=pipeline,
            X=np.zeros((2, 2)),
            y=np.array([0, 1]),
            threshold=1.5,
        )

    pipeline.predict_proba.assert_not_called()