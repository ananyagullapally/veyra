import os
from unittest.mock import patch

from veyra.orchestration import config


def test_retraining_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(config.os, "getenv", wraps=os.getenv):
            assert os.getenv(
                "VEYRA_RETRAINING_ENABLED", "false"
            ).lower() == "false"


def test_retraining_config_defaults():
    settings = config.RetrainingConfig()

    assert settings.enabled is False
    assert settings.check_interval_hours == 24
    assert settings.trigger_on_drift is True
    assert settings.minimum_production_predictions == 100
    assert settings.cooldown_hours == 168