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
Kafka: veyra.applications
        |
        v
Fraud Scoring Consumer
        |
        v
MLflow Champion Model
        |
        v
Kafka: veyra.predictions
        |
        v
Prediction Storage Consumer
        |
        v
PostgreSQL: fraud_predictions
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

### Start Local Infrastructure

Start Kafka and PostgreSQL:

    docker compose up -d

Start MLflow from the project root in a separate terminal:

    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

MLflow UI: http://127.0.0.1:5000

Kafka topics must be created explicitly because automatic topic creation is disabled.

### Start Fraud Scoring Services

Start each service in a separate terminal after Kafka, PostgreSQL, and MLflow are running.

**FastAPI:**

    uvicorn veyra.serving.api:app --host 127.0.0.1 --port 8000

**Kafka scoring consumer:**

    python -m veyra.streaming.consumer

**Kafka prediction storage consumer:**

    python -u -m veyra.storage.consumer

**API health check:**

    curl http://127.0.0.1:8000/health

### Run Tests

```bash
python -m pytest tests/ -q
```

Latest verified test results:

- 142 passed
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
