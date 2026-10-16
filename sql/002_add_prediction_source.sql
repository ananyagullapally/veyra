ALTER TABLE fraud_predictions
ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'unknown';

UPDATE fraud_predictions
SET source = 'pipeline_test'
WHERE application_id IN ('794990', '794991');

UPDATE fraud_predictions
SET source = 'synthetic_test'
WHERE application_id IN (
    'test-001',
    'integration-test-002',
    'idempotency-test-002',
    'outage-test-001'
);

ALTER TABLE fraud_predictions
DROP CONSTRAINT IF EXISTS fraud_predictions_source_check;

ALTER TABLE fraud_predictions
ADD CONSTRAINT fraud_predictions_source_check
CHECK (
    source IN (
        'unknown',
        'production',
        'pipeline_test',
        'synthetic_test'
    )
);