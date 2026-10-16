from pathlib import Path

import mlflow.sklearn
import pandas as pd

from veyra.training.mlflow_setup import configure_mlflow
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)


def main():
    configure_mlflow()

    model_uri = "models:/veyra-fraud-detector@champion"
    print(f"Loading champion: {model_uri}")

    pipeline = mlflow.sklearn.load_model(model_uri)

    validation_df = load_split_data("validation")
    X_validation, _ = create_feature_target_split(validation_df)

    probabilities = pipeline.predict_proba(X_validation)[:, 1]

    output_path = Path("data/processed/drift_reference.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_df = pd.DataFrame({
        "fraud_probability": probabilities,
    })

    reference_df.to_parquet(output_path, index=False)

    print(f"Reference predictions: {len(reference_df):,}")
    print(f"Average probability: {reference_df['fraud_probability'].mean():.6f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()