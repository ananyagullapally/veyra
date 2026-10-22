from datetime import datetime, timedelta, timezone


def check_retraining_cooldown(
    last_retraining_at: datetime | None,
    cooldown_hours: int = 168,
    now: datetime | None = None,
) -> dict:
    """Check whether enough time has passed since retraining."""

    if cooldown_hours < 0:
        raise ValueError("cooldown_hours cannot be negative.")

    now = now or datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware.")

    if last_retraining_at is None:
        return {
            "allowed": True,
            "reason": "No previous retraining attempt.",
        }

    if last_retraining_at.tzinfo is None:
        raise ValueError(
            "last_retraining_at must be timezone-aware."
        )

    next_allowed_at = last_retraining_at + timedelta(
        hours=cooldown_hours
    )

    if now < next_allowed_at:
        return {
            "allowed": False,
            "reason": "Retraining cooldown is active.",
            "next_allowed_at": next_allowed_at,
        }

    return {
        "allowed": True,
        "reason": "Retraining cooldown has elapsed.",
        "next_allowed_at": next_allowed_at,
    }