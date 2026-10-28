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

## Engineering Challenges & Solutions

### 1. Data Quality and Temporal Leakage

**Bottleneck:** The BAF dataset contained sentinel-coded missing values that could be misinterpreted as real measurements. For example, `prev_address_months_count` had 712,920 such entries, and `bank_months_count` had 253,635. Random splitting also risked temporal leakage.

**Solution:** Converted sentinel values to missing values, removed the constant `device_fraud_count` feature, validated schemas, and implemented chronological train/validation/test splits.

**Outcome:** Produced reproducible datasets containing 794,989 training, 108,168 validation, and 96,843 test applications.

### 2. Kafka Consumer Coordination and Reliable Persistence

**Bottleneck:** Connecting Kafka scoring, prediction events, and PostgreSQL introduced potential failures from duplicate deliveries, consumer interruptions, and unassigned partitions. During testing, two storage consumers were running, but one had no partition assignment.

**Solution:** Separated scoring and storage consumers, implemented retry handling and dead-letter routing, used manual Kafka offset management, and added duplicate-safe PostgreSQL writes. Inspected consumer assignments and terminated the redundant process.

**Outcome:** Verified that a fresh application was scored through the streaming pipeline and its prediction persisted in PostgreSQL.

### 3. MLflow Port Conflict and Champion Model Loading

**Bottleneck:** MLflow's default port `5000` conflicted with a macOS service, interfering with access to the registered champion model.

**Solution:** Moved MLflow to port `5001` and configured the tracking endpoint through the `MLFLOW_TRACKING_URI` environment variable.

**Outcome:** Confirmed FastAPI successfully loaded `veyra-fraud-detector` using the `champion` alias.

### 4. Monitoring Drift Without Immediate Fraud Labels

**Bottleneck:** Ground-truth fraud labels were not available from live prediction events, making immediate production accuracy monitoring impractical. Small samples could also produce misleading drift measurements.

**Solution:** Implemented Population Stability Index (PSI) monitoring of prediction-score distributions against a validation reference. Added thresholds for stable (<0.10), moderate (0.10–<0.25), and significant (>=0.25) drift. Required at least 100 production predictions and excluded pipeline-test records.

**Outcome:** Verified that monitoring correctly returned `insufficient_data` when there were not enough production predictions, instead of reporting an unreliable PSI.

### 5. Safe Retraining and Model Promotion

**Bottleneck:** Automatically retraining and replacing a champion model whenever drift is detected could cause overlapping jobs, excessive retraining, or deployment of an inferior candidate.

**Solution:** Added minimum sample requirements, retraining cooldowns, locking, candidate evaluation safeguards, and dry-run promotion. Kept automatic retraining disabled by default.

**Outcome:** Implemented controlled retraining orchestration without enabling unsafe automatic champion replacement.

### 6. End-to-End Integration Verification

**Bottleneck:** Passing unit tests or receiving a FastAPI prediction did not prove that Kafka consumers and PostgreSQL persistence worked together.

**Solution:** Added database integration tests, generated unique application IDs for demonstrations, inspected persisted prediction records, and separated test events from production monitoring data.

**Outcome:** Achieved 142 passing automated tests, 2 passing database integration tests, and verified end-to-end prediction persistence.

### Engineering Takeaway

The central challenge was not simply training XGBoost. It was integrating data processing, model serving, distributed event handling, reliable storage, monitoring, and retraining safeguards into a coherent MLOps workflow.

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
