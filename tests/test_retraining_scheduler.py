from unittest.mock import patch

from veyra.orchestration.config import RetrainingConfig
from veyra.orchestration.retraining_scheduler import (
    check_retraining_conditions,
)
from veyra.orchestration.retraining_scheduler import (
    check_retraining_conditions,
    run_scheduled_retraining,
)
@patch("veyra.orchestration.retraining_scheduler.RETRAINING_CONFIG")
@patch("veyra.orchestration.retraining_scheduler.check_prediction_drift")
def test_significant_drift_triggers_retraining(mock_drift, mock_config):
    mock_config.enabled = True
    mock_config.trigger_on_drift = True
    mock_config.minimum_production_predictions = 100

    mock_drift.return_value = {
        "status": "significant_drift",
        "psi": 0.30,
        "sample_count": 150,
        "minimum_required": 100,
    }

    result = check_retraining_conditions()

    assert result["status"] == "TRIGGERED"
    assert result["trigger"] is True
    assert result["psi"] == 0.30
    mock_drift.assert_called_once_with()


@patch("veyra.orchestration.retraining_scheduler.RETRAINING_CONFIG")
@patch("veyra.orchestration.retraining_scheduler.check_prediction_drift")
def test_stable_predictions_skip_retraining(mock_drift, mock_config):
    mock_config.enabled = True
    mock_config.trigger_on_drift = True
    mock_config.minimum_production_predictions = 100

    mock_drift.return_value = {
        "status": "stable",
        "psi": 0.05,
        "sample_count": 150,
        "minimum_required": 100,
    }

    result = check_retraining_conditions()

    assert result["status"] == "SKIPPED"
    assert result["trigger"] is False


@patch("veyra.orchestration.retraining_scheduler.RETRAINING_CONFIG")
@patch("veyra.orchestration.retraining_scheduler.check_prediction_drift")
def test_disabled_scheduler_skips_monitoring(mock_drift, mock_config):
    mock_config.enabled = False

    result = check_retraining_conditions()

    assert result["status"] == "SKIPPED"
    assert result["trigger"] is False
    mock_drift.assert_not_called()


@patch("veyra.orchestration.retraining_scheduler.RETRAINING_CONFIG")
@patch("veyra.orchestration.retraining_scheduler.check_prediction_drift")
def test_insufficient_data_skips_retraining(mock_drift, mock_config):
    mock_config.enabled = True
    mock_config.trigger_on_drift = True
    mock_config.minimum_production_predictions = 100

    mock_drift.return_value = {
        "status": "insufficient_data",
        "psi": None,
        "sample_count": 30,
        "minimum_required": 100,
    }

    result = check_retraining_conditions()

    assert result["status"] == "SKIPPED"
    assert result["trigger"] is False

@patch(
    "veyra.orchestration.retraining_scheduler.run_automated_retraining"
)
@patch(
    "veyra.orchestration.retraining_scheduler.check_retraining_conditions"
)
def test_scheduled_retraining_runs_when_triggered(
    mock_conditions,
    mock_retraining,
):
    mock_conditions.return_value = {
        "status": "TRIGGERED",
        "trigger": True,
    }

    mock_retraining.return_value = {
        "status": "COMPLETED",
        "action": "KEEP_CHAMPION",
    }

    result = run_scheduled_retraining()

    assert result["status"] == "COMPLETED"
    assert result["action"] == "KEEP_CHAMPION"

    mock_retraining.assert_called_once_with()


@patch(
    "veyra.orchestration.retraining_scheduler.run_automated_retraining"
)
@patch(
    "veyra.orchestration.retraining_scheduler.check_retraining_conditions"
)
def test_scheduled_retraining_skips_when_not_triggered(
    mock_conditions,
    mock_retraining,
):
    mock_conditions.return_value = {
        "status": "SKIPPED",
        "trigger": False,
        "reason": "No significant drift detected.",
    }

    result = run_scheduled_retraining()

    assert result["status"] == "SKIPPED"
    assert result["trigger"] is False

    mock_retraining.assert_not_called()