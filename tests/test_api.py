import pytest
from fastapi.testclient import TestClient
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)
import pandas as pd
import numpy as np
from veyra.serving.api import app


@pytest.fixture(scope="module")
def client():
    # Starts FastAPI and loads the MLflow champion model
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model"] == "veyra-fraud-detector"


def test_predict_rejects_missing_features(client):
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 422

    data = response.json()

    assert "missing_features" in data["detail"]


def test_predict_rejects_unexpected_features(client):
    response = client.post(
        "/predict",
        json={"features": {"unknown_feature": 123}},
    )

    assert response.status_code == 422

    data = response.json()

    assert "unknown_feature" in data["detail"]["unexpected_features"]


def test_predict_rejects_missing_request_body(client):
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422

def test_predict_rejects_invalid_numeric_value(client):
    validation_df = load_split_data("validation")
    X_validation, _ = create_feature_target_split(validation_df)

    # Select one validation application
    features = X_validation.iloc[0].to_dict()

    # Convert missing values to JSON null
    features = {
        key: (
            None if pd.isna(value)
            else value.item() if isinstance(value, np.generic)
            else value
        )
        for key, value in features.items()
    }

    # Deliberately corrupt a numeric feature
    features["income"] = "not_a_number"

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422