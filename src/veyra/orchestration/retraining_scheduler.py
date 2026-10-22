from veyra.monitoring.drift_monitor import check_prediction_drift
from veyra.orchestration.config import RETRAINING_CONFIG
from veyra.orchestration.retraining_pipeline import run_automated_retraining
from veyra.orchestration.retraining_trigger import (
    should_trigger_retraining,
)


def check_retraining_conditions() -> dict:
    """
    Check production drift and determine whether
    automated retraining should be requested.

    Does not execute model training.
    """

    if not RETRAINING_CONFIG.enabled:
        return {
            "status": "SKIPPED",
            "trigger": False,
            "reason": "Automated retraining is disabled.",
        }

    drift_result = check_prediction_drift()

    decision = should_trigger_retraining(
        drift_result=drift_result,
        config=RETRAINING_CONFIG,
    )

    return {
        "status": "TRIGGERED" if decision["trigger"] else "SKIPPED",
        "trigger": decision["trigger"],
        "reason": decision["reason"],
        "drift_status": drift_result["status"],
        "psi": drift_result["psi"],
        "sample_count": drift_result["sample_count"],
    }

def run_scheduled_retraining() -> dict:
    """
    Check drift conditions and initiate protected retraining
    only when the trigger requirements are satisfied.
    """
    decision = check_retraining_conditions()

    if not decision["trigger"]:
        return decision

    return run_automated_retraining()

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

if __name__ == "__main__":
    result = run_scheduled_retraining()
    print(result)