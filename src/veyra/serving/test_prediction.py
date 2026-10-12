import json
from urllib.request import Request, urlopen

import mlflow.sklearn
import numpy as np
import pandas as pd

from veyra.training.mlflow_setup import configure_mlflow
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)


def main():
    configure_mlflow()

    # Use one local validation record, not final test data
    validation_df = load_split_data("validation")
    X_validation, _ = create_feature_target_split(validation_df)

    sample = X_validation.head(1)

    # Convert pandas missing values to JSON null
    features = {
        key: None if pd.isna(value) else (
            value.item() if isinstance(value, np.generic) else value
        )
        for key, value in sample.iloc[0].items()
    }

    payload = json.dumps({"features": features}).encode("utf-8")

    request = Request(
        "http://127.0.0.1:8000/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request) as response:
        result = json.load(response)

    # Independently score the same record using MLflow
    pipeline = mlflow.sklearn.load_model(
        "models:/veyra-fraud-detector@champion"
    )

    expected_probability = float(
        pipeline.predict_proba(sample)[0, 1]
    )

    api_probability = result["fraud_probability"]

    assert np.isclose(
        api_probability,
        expected_probability,
        atol=1e-7,
    ), "API and MLflow predictions differ!"

    assert result["decision"] == (
        "manual_review"
        if api_probability >= result["threshold"]
        else "not_flagged"
    )

    print("API fraud probability:", round(api_probability, 6))
    print("MLflow probability:   ", round(expected_probability, 6))
    print("Decision:             ", result["decision"])
    print("Threshold:            ", result["threshold"])
    print("\nAPI prediction verified successfully!")


if __name__ == "__main__":
    main()