# Veyra — macOS Retraining Scheduler

This directory contains the macOS launchd configuration for Veyra's automated retraining scheduler.

## Architecture

The real-time fraud scoring pipeline is:

Banking Application → Kafka Applications → Scoring Consumer → FastAPI /predict → Kafka Predictions → Storage Consumer → PostgreSQL

PostgreSQL prediction records are used for prediction-score PSI monitoring.

When retraining is enabled and its trigger conditions are met, Veyra applies a PostgreSQL lock and cooldown before candidate training and evaluation.

Model promotion is dry-run only. The registered MLflow champion is not automatically replaced.

## Scheduler

The `com.veyra.retraining.plist` file configures the macOS launchd scheduler.

Automated retraining is disabled by default. The scheduler requires the appropriate local environment and services to be available.

For configuration and safety details, see [Automated Retraining Documentation](../../docs/automated_retraining.md).

For the complete system architecture, see the [Project README](../../README.md).
