# Veyra — Real-Time Fraud Detection & MLOps Platform

Veyra is an end-to-end fraud detection platform combining machine learning, real-time event streaming, model serving, monitoring, and retraining orchestration.

## Technology Stack

- Python, XGBoost, scikit-learn
- MLflow for experiment tracking
- FastAPI for model serving
- Apache Kafka for event streaming
- PostgreSQL for prediction storage
- PSI for prediction drift monitoring
- pytest for automated testing

## Key Features

- Temporal train, validation, and test splitting
- Real-time fraud scoring
- Kafka-based prediction processing
- PostgreSQL prediction persistence
- Production drift monitoring
- Candidate model training and evaluation
- Champion-versus-candidate comparison
- Retraining cooldown and concurrency protection
- Dry-run model promotion

## Automated Retraining

Retraining is disabled by default. When enabled, the scheduled workflow checks production drift before initiating candidate training.

Model promotion remains dry-run only.

See [Retraining Documentation](docs/automated_retraining.md) and [Scheduler Setup](deployment/launchd/README.md).

## System Architecture

```text
Banking Application
        |
        v
Apache Kafka (veyra.applications)
        |
        v
Fraud Scoring Consumer
        |
        +----> MLflow Champion Model
        |
        +----> PostgreSQL Prediction Storage
        |
        +----> Kafka Predictions Topic
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

FastAPI provides `/health` and `/predict` endpoints for synchronous fraud scoring.

Automated retraining is disabled by default. Candidate promotion does not modify the production champion.

## Local Development

### Prerequisites

- Python 3.11
- Docker and Docker Compose
- PostgreSQL
- Apache Kafka
- MLflow

### Environment Setup

Clone the repository:

```bash
git clone https://github.com/ananyagullapally/veyra.git
cd veyra
```

Create a Python environment and install the project:

```bash
conda create -n veyra python=3.11
conda activate veyra
pip install -e .
```

Configure environment variables using `.env.example` as a reference.

### Run Tests

```bash
python -m pytest tests/ -q
```

Latest verified test results:

- 140 passed
- 3 skipped
- 219 warnings

### Run the Retraining Scheduler

```bash
python -m veyra.orchestration.retraining_scheduler
```

Automated retraining is disabled by default.

To inspect the macOS scheduler configuration, see
[launchd Setup](deployment/launchd/README.md).

## Model Safety

Candidate models are evaluated against the registered MLflow champion.

Promotion decisions are dry-run only. Veyra does not automatically
replace the production champion.
