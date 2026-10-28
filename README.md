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

## Dataset & Model Evaluation

Veyra uses the Bank Account Fraud (BAF) dataset, containing 1 million privacy-preserving synthetic bank account applications. BAF was selected over simpler synthetic datasets because it provides application-specific banking, customer, and device signals, realistic class imbalance, and a temporal structure suitable for evaluating fraud detection.

The preprocessing pipeline handles missing-value sentinels, validates feature schemas, and removes constant features.

Data is split chronologically:

- Training: 794,989 applications
- Validation: 108,168 applications
- Testing: 96,843 applications

An XGBoost classifier produces fraud probabilities, with a configurable decision threshold. MLflow tracks experiments and manages the registered champion model.

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

    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001

MLflow UI: http://127.0.0.1:5001

Veyra defaults to MLflow at port 5001. To use another tracking server, set `MLFLOW_TRACKING_URI` before starting FastAPI or the scoring consumer, for example: `export MLFLOW_TRACKING_URI=http://127.0.0.1:5001`.

Kafka topics must be created explicitly because automatic topic creation is disabled.

Create the required topics after starting Docker Compose:

```bash
for topic in veyra.applications veyra.predictions veyra.applications.dlq; do
  docker exec veyra-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic "$topic" --partitions 1 --replication-factor 1
done
```

These settings are intended for local development with a single Kafka broker.

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

## Portfolio Case Study

Read the [Veyra Portfolio Case Study](docs/portfolio_case_study.md) for the problem statement, dataset selection, technical architecture, engineering challenges, and verified outcomes.
