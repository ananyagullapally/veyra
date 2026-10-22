import pytest

from veyra.training.compare_models import compare_models


def test_candidate_eligible_for_promotion():
    champion = {
        "pr_auc": 0.22,
        "precision": 0.07,
        "recall": 0.79,
    }

    candidate = {
        "pr_auc": 0.25,
        "precision": 0.09,
        "recall": 0.85,
    }

    result = compare_models(champion, candidate)

    assert result["eligible_for_promotion"] is True
    assert result["passes_recall"] is True
    assert result["improves_pr_auc"] is True
    assert result["pr_auc_improvement"] == pytest.approx(0.03)


def test_candidate_rejected_for_low_recall():
    champion = {
        "pr_auc": 0.22,
        "precision": 0.07,
        "recall": 0.79,
    }

    candidate = {
        "pr_auc": 0.26,
        "precision": 0.10,
        "recall": 0.70,
    }

    result = compare_models(champion, candidate)

    assert result["eligible_for_promotion"] is False
    assert result["passes_recall"] is False


def test_candidate_rejected_for_lower_pr_auc():
    champion = {
        "pr_auc": 0.22,
        "precision": 0.07,
        "recall": 0.79,
    }

    candidate = {
        "pr_auc": 0.20,
        "precision": 0.09,
        "recall": 0.85,
    }

    result = compare_models(champion, candidate)

    assert result["eligible_for_promotion"] is False
    assert result["improves_pr_auc"] is False


def test_missing_metric_raises_error():
    champion = {
        "pr_auc": 0.22,
        "precision": 0.07,
        "recall": 0.79,
    }

    candidate = {
        "pr_auc": 0.25,
        "recall": 0.85,
    }

    with pytest.raises(ValueError, match="precision"):
        compare_models(champion, candidate)