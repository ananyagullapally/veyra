# Veyra — Real-Time Fraud Detection & MLOps Platform

Veyra is an end-to-end machine learning platform for detecting fraudulent banking applications using real-time event streaming, model serving, experiment tracking, production monitoring, and automated retraining orchestration.

## Technology Stack

- **Machine Learning:** Python, XGBoost, scikit-learn
- **Experiment Tracking:** MLflow
- **API:** FastAPI
- **Event Streaming:** Apache Kafka
- **Database:** PostgreSQL
- **Monitoring:** Population Stability Index (PSI)
- **Orchestration:** Python, PostgreSQL advisory locks, macOS launchd
- **Testing:** pytest

## Key Capabilities

- Temporal data splitting to reduce data leakage
- Fraud prediction through a FastAPI endpoint
- Kafka-based application ingestion and prediction publishing
- PostgreSQL persistence of prediction results
- Production prediction drift monitoring
- Automated candidate training and evaluation
- Champion-versus-candidate model comparison
- Dry-run promotion with safety controls
- Drift-triggered retraining with concurrency protection and cooldown

## Automated Retraining

Veyra supports drift-triggered retraining orchestration, with a seven-day cooldown and PostgreSQL advisory locking.

Retraining is disabled by default, and automatic champion promotion is not implemented.

See [Automated Retraining Documentation](docs/automated_retraining.md) and [macOS Scheduler Setup](deployment/launchd/README.md).

## System Architecture

```text
Banking Application
        |
        v
Apache Kafka
(veyra.applications)
        |
        v
Fraud Scoring Consumer
        |
        +----> MLflow Champion Model
        |             |
        |             v
        |       Fraud Probability
        |             |
        v             v
PostgreSQL      Kafka Predictions
        |
        v
PSI Drift Monitoring
        |
        v
Retraining Trigger
        |
        v
PostgreSQL Lock + Cooldown
        |
        v
XGBoost Candidate Training
        |
        v
MLflow Candidate Evaluation
        |
        v
Dry-Run Promotion Decision
```

The FastAPI service also provides `/health` and `/predict` endpoints for synchronous model serving.

Automated retraining is disabled by default, and candidate promotion does not modify the production champion.