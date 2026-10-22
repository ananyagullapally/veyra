from veyra.training.promote_candidate import promote_candidate
from veyra.training.train_candidate import main as train_candidate
from veyra.orchestration.retraining_lock import acquire_retraining_lock
from veyra.orchestration.config import RETRAINING_CONFIG
from veyra.orchestration.retraining_history import get_last_retraining_at
from veyra.orchestration.retraining_cooldown import check_retraining_cooldown
from veyra.orchestration.retraining_history import (
    start_retraining_attempt,
    complete_retraining_attempt,
    fail_retraining_attempt,
)
def run_retraining_pipeline(candidate_run_id: str) -> dict:
    """
    Evaluate a trained candidate and produce a
    safe, dry-run promotion decision.
    """

    result = promote_candidate(
        run_id=candidate_run_id,
        dry_run=True,
    )

    return {
        "candidate_run_id": candidate_run_id,
        "champion_version": result["champion_version"],
        "eligible_for_promotion": result["eligible_for_promotion"],
        "action": result["action"],
        "status": "COMPLETED",
    }

def run_automated_retraining() -> dict:
    """
    Train and evaluate a candidate while preventing
    overlapping jobs and recording retraining history.
    """
    with acquire_retraining_lock():
        last_retraining_at = get_last_retraining_at()

        cooldown = check_retraining_cooldown(
            last_retraining_at=last_retraining_at,
            cooldown_hours=RETRAINING_CONFIG.cooldown_hours,
        )

        if not cooldown["allowed"]:
            return {
                "status": "SKIPPED",
                "action": "COOLDOWN_ACTIVE",
                "reason": cooldown["reason"],
                "next_allowed_at": cooldown["next_allowed_at"],
            }

        attempt_id = start_retraining_attempt()

        try:
            candidate_run_id = train_candidate()

            if not candidate_run_id:
                raise ValueError(
                    "Candidate training did not return an MLflow run ID."
                )

            result = run_retraining_pipeline(
                candidate_run_id=candidate_run_id
            )

            complete_retraining_attempt(
                attempt_id=attempt_id,
                candidate_run_id=candidate_run_id,
                action=result["action"],
            )

            return result

        except Exception as error:
            fail_retraining_attempt(
                attempt_id=attempt_id,
                error_message=str(error),
            )
            raise