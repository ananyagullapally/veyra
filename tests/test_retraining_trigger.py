import pytest

from veyra.orchestration.config import RetrainingConfig
from veyra.orchestration.retraining_trigger import (
    should_trigger_retraining,
)


@pytest.fixture
def config():
    return RetrainingConfig(
        enabled=True,
        trigger_on_drift=True,
        minimum_production_predictions=100,
    )


@pytest.fixture
def drift_result():
    return {
        "status": "significant_drift",
        "sample_count": 150,
        "minimum_required": 100,
        "psi": 0.30,
    }


def test_significant_drift_triggers_retraining(config, drift_result):
    result = should_trigger_retraining(drift_result, config)
    assert result["trigger"] is True


def test_disabled_retraining(drift_result):
    config = RetrainingConfig(enabled=False)
    result = should_trigger_retraining(drift_result, config)
    assert result["trigger"] is False


def test_insufficient_samples(config, drift_result):
    drift_result["sample_count"] = 50
    result = should_trigger_retraining(drift_result, config)
    assert result["trigger"] is False


@pytest.mark.parametrize(
    "status",
    ["stable", "moderate_drift", "insufficient_data"],
)
def test_other_statuses_do_not_trigger(config, drift_result, status):
    drift_result["status"] = status
    result = should_trigger_retraining(drift_result, config)
    assert result["trigger"] is False


def test_drift_trigger_disabled(drift_result):
    config = RetrainingConfig(
        enabled=True,
        trigger_on_drift=False,
    )
    result = should_trigger_retraining(drift_result, config)
    assert result["trigger"] is False


def test_monitor_minimum_is_respected(config, drift_result):
    drift_result["minimum_required"] = 200
    result = should_trigger_retraining(drift_result, config)
    assert result["trigger"] is False