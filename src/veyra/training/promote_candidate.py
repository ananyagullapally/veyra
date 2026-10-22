from veyra.training.evaluate_candidate import evaluate_candidate


def promote_candidate(run_id: str, dry_run: bool = True) -> dict:
    if not dry_run:
        raise NotImplementedError(
            "Live model promotion is not enabled yet."
        )

    result = evaluate_candidate(run_id)
    comparison = result["comparison"]

    eligible = comparison["eligible_for_promotion"]

    return {
        "candidate_run_id": run_id,
        "champion_version": result["champion_version"],
        "eligible_for_promotion": eligible,
        "dry_run": True,
        "action": (
            "WOULD_PROMOTE" if eligible else "KEEP_CHAMPION"
        ),
        "reason": (
            "Candidate passed promotion requirements."
            if eligible
            else "Candidate did not meet promotion requirements."
        ),
    }