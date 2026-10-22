import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainingConfig:
    enabled: bool = False
    check_interval_hours: int = 24
    trigger_on_drift: bool = True
    minimum_production_predictions: int = 100
    cooldown_hours: int = 168


RETRAINING_CONFIG = RetrainingConfig(
    enabled=os.getenv(
        "VEYRA_RETRAINING_ENABLED", "false"
    ).lower() == "true"
)