from unittest.mock import patch

import pytest

from veyra.training.promote_candidate import promote_candidate


@patch("veyra.training.promote_candidate.evaluate_candidate")
def test_rejects_candidate_without_improvement(mock_evaluate):
    mock_evaluate.return_value = {
        "champion_version": "1",
        "comparison": {
            "eligible_for_promotion": False,
        },
    }

    result = promote_candidate("candidate-123")

    assert result["action"] == "KEEP_CHAMPION"
    assert result["eligible_for_promotion"] is False
    assert result["dry_run"] is True


@patch("veyra.training.promote_candidate.evaluate_candidate")
def test_eligible_candidate_still_uses_dry_run(mock_evaluate):
    mock_evaluate.return_value = {
        "champion_version": "1",
        "comparison": {
            "eligible_for_promotion": True,
        },
    }

    result = promote_candidate("candidate-123")

    assert result["action"] == "WOULD_PROMOTE"
    assert result["dry_run"] is True


@patch("veyra.training.promote_candidate.evaluate_candidate")
def test_live_promotion_is_blocked(mock_evaluate):
    with pytest.raises(NotImplementedError):
        promote_candidate("candidate-123", dry_run=False)

    mock_evaluate.assert_not_called()