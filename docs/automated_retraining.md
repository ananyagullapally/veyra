# Automated Retraining & Orchestration

## Overview

Veyra supports drift-triggered retraining with safeguards for concurrency, cooldown enforcement, and model evaluation.

## Workflow

1. Monitor production prediction scores using PSI.
2. Trigger retraining when significant drift is detected.
3. Acquire a PostgreSQL advisory lock.
4. Enforce a seven-day retraining cooldown.
5. Train an XGBoost candidate model.
6. Log candidate metrics and artifacts to MLflow.
7. Compare the candidate against the champion.
8. Record retraining outcomes in PostgreSQL.

## Promotion Criteria

A candidate is eligible when:

- Validation recall is at least 0.80.
- Validation PR-AUC exceeds the champion's validation PR-AUC.

Promotion is dry-run only. The production champion is not automatically replaced.

## Configuration

Retraining is disabled by default.

Environment variable:

`VEYRA_RETRAINING_ENABLED=false`

## Scheduling

A macOS launchd configuration runs the scheduler every 24 hours while the development Mac and user session are available.

See `deployment/launchd/README.md`.

## Limitations

- Candidate training currently reuses the existing labeled dataset.
- Newly labeled production outcomes are not automatically incorporated.
- Production champion promotion is not automated.
- The scheduler runs locally rather than in the cloud.
- End-to-end retraining under real production drift has not yet been validated.
