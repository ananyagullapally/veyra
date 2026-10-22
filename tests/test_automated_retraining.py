from contextlib import nullcontext
from unittest.mock import patch

import pytest

from veyra.orchestration.retraining_pipeline import (
    run_automated_retraining,
)


@patch(
    "veyra.orchestration.retraining_pipeline.acquire_retraining_lock",
    return_value=nullcontext(),
)
@patch("veyra.orchestration.retraining_pipeline.start_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.complete_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.fail_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.run_retraining_pipeline")
@patch("veyra.orchestration.retraining_pipeline.train_candidate")
@patch(
    "veyra.orchestration.retraining_pipeline.get_last_retraining_at",
    return_value=None,
)
@patch(
    "veyra.orchestration.retraining_pipeline.check_retraining_cooldown",
    return_value={"allowed": True},
)
def test_automated_retraining(
    mock_cooldown,
    mock_last_retraining,
    mock_train,
    mock_pipeline,
    mock_fail,
    mock_complete,
    mock_start,
    mock_lock,
):
    mock_start.return_value = 42
    mock_train.return_value = "candidate-123"
    mock_pipeline.return_value = {
        "candidate_run_id": "candidate-123",
        "action": "KEEP_CHAMPION",
        "status": "COMPLETED",
    }

    result = run_automated_retraining()

    assert result["status"] == "COMPLETED"
    assert result["action"] == "KEEP_CHAMPION"

    mock_start.assert_called_once_with()
    mock_train.assert_called_once_with()
    mock_pipeline.assert_called_once_with(
        candidate_run_id="candidate-123"
    )
    mock_complete.assert_called_once_with(
        attempt_id=42,
        candidate_run_id="candidate-123",
        action="KEEP_CHAMPION",
    )
    mock_fail.assert_not_called()


@patch(
    "veyra.orchestration.retraining_pipeline.acquire_retraining_lock",
    return_value=nullcontext(),
)
@patch("veyra.orchestration.retraining_pipeline.start_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.complete_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.fail_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.run_retraining_pipeline")
@patch("veyra.orchestration.retraining_pipeline.train_candidate")
@patch(
    "veyra.orchestration.retraining_pipeline.get_last_retraining_at",
    return_value=None,
)
@patch(
    "veyra.orchestration.retraining_pipeline.check_retraining_cooldown",
    return_value={"allowed": True},
)
def test_missing_candidate_run_id(
    mock_cooldown,
    mock_last_retraining,
    mock_train,
    mock_pipeline,
    mock_fail,
    mock_complete,
    mock_start,
    mock_lock,
):
    mock_start.return_value = 42
    mock_train.return_value = None

    with pytest.raises(ValueError, match="MLflow run ID"):
        run_automated_retraining()

    mock_complete.assert_not_called()
    mock_pipeline.assert_not_called()
    mock_fail.assert_called_once_with(
        attempt_id=42,
        error_message="Candidate training did not return an MLflow run ID.",
    )

@patch(
    "veyra.orchestration.retraining_pipeline.acquire_retraining_lock",
    return_value=nullcontext(),
)
@patch("veyra.orchestration.retraining_pipeline.start_retraining_attempt")
@patch("veyra.orchestration.retraining_pipeline.train_candidate")
@patch(
    "veyra.orchestration.retraining_pipeline.get_last_retraining_at"
)
@patch(
    "veyra.orchestration.retraining_pipeline.check_retraining_cooldown"
)
def test_active_cooldown_skips_retraining(
    mock_cooldown,
    mock_last_retraining,
    mock_train,
    mock_start,
    mock_lock,
):
    from datetime import datetime, timezone

    last_attempt = datetime(2026, 10, 20, tzinfo=timezone.utc)
    next_allowed = datetime(2026, 10, 27, tzinfo=timezone.utc)

    mock_last_retraining.return_value = last_attempt
    mock_cooldown.return_value = {
        "allowed": False,
        "reason": "Retraining cooldown is active.",
        "next_allowed_at": next_allowed,
    }

    result = run_automated_retraining()

    assert result["status"] == "SKIPPED"
    assert result["action"] == "COOLDOWN_ACTIVE"
    assert result["next_allowed_at"] == next_allowed

    mock_train.assert_not_called()
    mock_start.assert_not_called()

    mock_cooldown.assert_called_once_with(
        last_retraining_at=last_attempt,
        cooldown_hours=168,
    )