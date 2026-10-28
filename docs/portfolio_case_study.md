# Veyra — Real-Time Fraud Detection & MLOps

## Problem Statement
Fraud detection requires more than training a machine learning model. A practical system must process application events, generate predictions, persist results reliably, monitor changing model behavior, and manage retraining safely.

Veyra demonstrates an end-to-end, production-oriented fraud detection pipeline.

## Dataset Selection
The project uses the Bank Account Fraud (BAF) dataset containing 1 million privacy-preserving synthetic financial applications.

BAF provides realistic application features, class imbalance, and temporal structure, making it suitable for chronological evaluation and streaming simulation.

## Technical Architecture
- Python and XGBoost for fraud classification
- MLflow for experiment tracking and champion model management
- Apache Kafka for event-driven processing
- FastAPI for model inference
- PostgreSQL for prediction persistence
- Docker for local infrastructure
- PSI for prediction-score drift monitoring

## Engineering Challenges

### Data Quality and Leakage
Handled missing-value sentinels, validated schemas, and implemented chronological train/validation/test splits.

### Streaming Reliability
Implemented retry handling, dead-letter queues, manual Kafka offset management, and duplicate-safe PostgreSQL writes.

### Model Lifecycle Safety
Implemented minimum monitoring sample requirements, PSI thresholds, retraining cooldowns, locking, candidate evaluation, and dry-run promotion.

### MLflow Integration
Resolved a local port conflict by configuring the MLflow tracking endpoint through environment variables.

## Verified Outcomes
- Processed 1,000,000 application records.
- Built and verified the Kafka-to-PostgreSQL prediction pipeline.
- Registered and served an XGBoost champion model.
- Passed 142 automated tests and 2 database integration tests.
- Implemented PSI-based prediction-score monitoring and controlled retraining safeguards.

## Key Takeaway
Veyra demonstrates how machine learning can be integrated with streaming infrastructure, reliable storage, monitoring, and model lifecycle management.

## Repository
https://github.com/ananyagullapally/veyra
