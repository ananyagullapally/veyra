CREATE TABLE IF NOT EXISTS fraud_predictions (
    application_id TEXT PRIMARY KEY,
    fraud_probability DOUBLE PRECISION NOT NULL
        CHECK (fraud_probability BETWEEN 0 AND 1),
    decision TEXT NOT NULL
        CHECK (decision IN ('flagged', 'not_flagged')),
    threshold DOUBLE PRECISION NOT NULL
        CHECK (threshold BETWEEN 0 AND 1),
    model TEXT NOT NULL,
    alias TEXT NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
