from veyra.orchestration.config import (
    RETRAINING_CONFIG,
    RetrainingConfig,
)


def should_trigger_retraining(
    drift_result: dict,
    config: RetrainingConfig = RETRAINING_CONFIG,
) -> dict:
    """Determine whether drift justifies a retraining request."""

    if not config.enabled:
        return {
            "trigger": False,
            "reason": "Automated retraining is disabled.",
        }

    if not config.trigger_on_drift:
        return {
            "trigger": False,
            "reason": "Drift-based retraining is disabled.",
        }

    sample_count = drift_result["sample_count"]

    minimum_required = max(
        config.minimum_production_predictions,
        drift_result["minimum_required"],
    )

    if sample_count < minimum_required:
        return {
            "trigger": False,
            "reason": "Insufficient production predictions.",
        }

    if drift_result["status"] != "significant_drift":
        return {
            "trigger": False,
            "reason": "Significant drift not detected.",
        }

    return {
        "trigger": True,
        "reason": "Significant prediction drift detected.",
    }