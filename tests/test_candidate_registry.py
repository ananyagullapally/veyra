from types import SimpleNamespace
from unittest.mock import patch

import pytest

from veyra.training.candidate_registry import get_candidate_info


RUN_ID = "test-candidate-run"


def make_run(threshold="0.25"):
    return SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "validation_pr_auc": 0.20,
                "validation_precision": 0.10,
                "validation_recall": 0.85,
            },
            params={"selected_threshold": threshold},
        )
    )


@patch("veyra.training.candidate_registry.MlflowClient")
@patch("veyra.training.candidate_registry.configure_mlflow")
def test_valid_candidate(mock_configure, mock_client):
    mock_client.return_value.get_run.return_value = make_run()

    result = get_candidate_info(RUN_ID)

    assert result["run_id"] == RUN_ID
    assert result["metrics"]["pr_auc"] == 0.20
    assert result["metrics"]["recall"] == 0.85
    assert result["threshold"] == 0.25


@pytest.mark.parametrize(
    "threshold",
    [None, "invalid", "-0.1", "1.5", "nan", "inf", "-inf"],
)
@patch("veyra.training.candidate_registry.MlflowClient")
@patch("veyra.training.candidate_registry.configure_mlflow")
def test_invalid_threshold(mock_configure, mock_client, threshold):
    mock_client.return_value.get_run.return_value = make_run(threshold)

    with pytest.raises(ValueError):
        get_candidate_info(RUN_ID)


@patch("veyra.training.candidate_registry.MlflowClient")
@patch("veyra.training.candidate_registry.configure_mlflow")
def test_missing_metric(mock_configure, mock_client):
    run = make_run()
    del run.data.metrics["validation_pr_auc"]
    mock_client.return_value.get_run.return_value = run

    with pytest.raises(ValueError, match="validation_pr_auc"):
        get_candidate_info(RUN_ID)