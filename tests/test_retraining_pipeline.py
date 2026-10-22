from unittest.mock import patch

from veyra.orchestration.retraining_pipeline import (
    run_retraining_pipeline,
)


@patch(
    "veyra.orchestration.retraining_pipeline.promote_candidate"
)
def test_pipeline_keeps_champion(mock_promote):
    mock_promote.return_value = {
        "champion_version": "1",
        "eligible_for_promotion": False,
        "action": "KEEP_CHAMPION",
    }

    result = run_retraining_pipeline("candidate-123")

    assert result["status"] == "COMPLETED"
    assert result["action"] == "KEEP_CHAMPION"
    assert result["eligible_for_promotion"] is False

    mock_promote.assert_called_once_with(
        run_id="candidate-123",
        dry_run=True,
    )


@patch(
    "veyra.orchestration.retraining_pipeline.promote_candidate"
)
def test_pipeline_reports_eligible_candidate(mock_promote):
    mock_promote.return_value = {
        "champion_version": "1",
        "eligible_for_promotion": True,
        "action": "WOULD_PROMOTE",
    }

    result = run_retraining_pipeline("candidate-456")

    assert result["status"] == "COMPLETED"
    assert result["action"] == "WOULD_PROMOTE"
    assert result["eligible_for_promotion"] is True

    mock_promote.assert_called_once_with(
        run_id="candidate-456",
        dry_run=True,
    )