from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from veyra.training.evaluate_candidate import evaluate_candidate


@patch("veyra.training.evaluate_candidate.compare_models")
@patch("veyra.training.evaluate_candidate.evaluate_model")
@patch("mlflow.sklearn.load_model")
@patch("veyra.training.evaluate_candidate.MlflowClient")
@patch("veyra.training.evaluate_candidate.create_feature_target_split")
@patch("veyra.training.evaluate_candidate.load_split_data")
@patch("veyra.training.evaluate_candidate.get_candidate_info")
@patch("veyra.training.evaluate_candidate.get_champion_info")
@patch("veyra.training.evaluate_candidate.configure_mlflow")
def test_candidate_without_improvement(
    mock_configure,
    mock_champion,
    mock_candidate,
    mock_load,
    mock_split,
    mock_client,
    mock_model,
    mock_evaluate,
    mock_compare,
):
    mock_champion.return_value = {
        "version": "1",
        "model_uri": "models:/veyra-fraud-detector@champion",
        "threshold": 0.0132,
    }

    mock_candidate.return_value = {
        "run_id": "candidate-123",
        "threshold": 0.0132,
        "metrics": {},
    }

    mock_load.return_value = "validation_data"
    mock_split.return_value = ("features", "labels")

    mock_client.return_value.search_logged_models.return_value = [
        SimpleNamespace(model_id="candidate-model", status="READY")
    ]

    mock_model.return_value = MagicMock(predict_proba=MagicMock())

    metrics = {
        "pr_auc": 0.18345,
        "precision": 0.05936,
        "recall": 0.8,
    }
    mock_evaluate.return_value = metrics

    mock_compare.return_value = {
        "eligible_for_promotion": False,
        "pr_auc_improvement": 0.0,
    }

    result = evaluate_candidate("candidate-123")

    assert result["champion_version"] == "1"
    assert result["comparison"]["eligible_for_promotion"] is False

    assert mock_evaluate.call_count == 2
    mock_compare.assert_called_once_with(
        champion_metrics=metrics,
        candidate_metrics=metrics,
    )

@patch("veyra.training.evaluate_candidate.MlflowClient")
@patch("veyra.training.evaluate_candidate.get_candidate_info")
@patch("veyra.training.evaluate_candidate.get_champion_info")
@patch("veyra.training.evaluate_candidate.configure_mlflow")
def test_rejects_missing_candidate_artifact(
    mock_configure,
    mock_champion,
    mock_candidate,
    mock_client,
):
    import pytest

    mock_champion.return_value = {
        "version": "1",
        "model_uri": "models:/veyra-fraud-detector@champion",
        "threshold": 0.0132,
    }

    mock_candidate.return_value = {
        "run_id": "candidate-123",
        "threshold": 0.0132,
        "metrics": {},
    }

    mock_client.return_value.search_logged_models.return_value = []

    with pytest.raises(ValueError, match="exactly one READY"):
        evaluate_candidate("candidate-123")
@patch("veyra.training.evaluate_candidate.MlflowClient")
@patch("veyra.training.evaluate_candidate.get_candidate_info")
@patch("veyra.training.evaluate_candidate.get_champion_info")
@patch("veyra.training.evaluate_candidate.configure_mlflow")
def test_rejects_multiple_ready_artifacts(
    mock_configure,
    mock_champion,
    mock_candidate,
    mock_client,
):
    import pytest

    mock_champion.return_value = {
        "version": "1",
        "model_uri": "models:/veyra-fraud-detector@champion",
        "threshold": 0.0132,
    }

    mock_candidate.return_value = {
        "run_id": "candidate-123",
        "threshold": 0.0132,
        "metrics": {},
    }

    mock_client.return_value.search_logged_models.return_value = [
        SimpleNamespace(model_id="model-1", status="READY"),
        SimpleNamespace(model_id="model-2", status="READY"),
    ]

    with pytest.raises(ValueError, match="exactly one READY"):
        evaluate_candidate("candidate-123")

@patch("veyra.training.evaluate_candidate.MlflowClient")
@patch("veyra.training.evaluate_candidate.get_candidate_info")
@patch("veyra.training.evaluate_candidate.get_champion_info")
@patch("veyra.training.evaluate_candidate.configure_mlflow")
def test_rejects_failed_candidate_artifact(
    mock_configure,
    mock_champion,
    mock_candidate,
    mock_client,
):
    import pytest

    mock_champion.return_value = {
        "version": "1",
        "model_uri": "models:/veyra-fraud-detector@champion",
        "threshold": 0.0132,
    }

    mock_candidate.return_value = {
        "run_id": "candidate-123",
        "threshold": 0.0132,
        "metrics": {},
    }

    mock_client.return_value.search_logged_models.return_value = [
        SimpleNamespace(model_id="failed-model", status="FAILED")
    ]

    with pytest.raises(ValueError, match="exactly one READY"):
        evaluate_candidate("candidate-123")