from datetime import datetime, timedelta, timezone

import pytest

from veyra.orchestration.retraining_cooldown import (
    check_retraining_cooldown,
)


NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def test_no_previous_retraining():
    result = check_retraining_cooldown(
        last_retraining_at=None,
        now=NOW,
    )

    assert result["allowed"] is True


def test_active_cooldown():
    result = check_retraining_cooldown(
        last_retraining_at=NOW - timedelta(hours=24),
        cooldown_hours=168,
        now=NOW,
    )

    assert result["allowed"] is False
    assert result["next_allowed_at"] == NOW + timedelta(hours=144)


def test_expired_cooldown():
    result = check_retraining_cooldown(
        last_retraining_at=NOW - timedelta(hours=200),
        cooldown_hours=168,
        now=NOW,
    )

    assert result["allowed"] is True


def test_exact_cooldown_boundary():
    result = check_retraining_cooldown(
        last_retraining_at=NOW - timedelta(hours=168),
        cooldown_hours=168,
        now=NOW,
    )

    assert result["allowed"] is True


def test_negative_cooldown_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        check_retraining_cooldown(
            last_retraining_at=None,
            cooldown_hours=-1,
            now=NOW,
        )


def test_naive_last_retraining_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        check_retraining_cooldown(
            last_retraining_at=datetime(2026, 10, 15),
            now=NOW,
        )


def test_naive_current_time_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        check_retraining_cooldown(
            last_retraining_at=None,
            now=datetime(2026, 10, 16),
        )


def test_zero_cooldown():
    result = check_retraining_cooldown(
        last_retraining_at=NOW,
        cooldown_hours=0,
        now=NOW,
    )

    assert result["allowed"] is True