import mlflow.sklearn

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

    X_validation, _ = create_feature_target_split(
        validation_df
    )

    sample = X_validation.head(5)

    probabilities = pipeline.predict_proba(sample)[:, 1]

    print("\nChampion fraud probabilities:")
    print(probabilities)

    print("\nChampion model loaded successfully!")


if __name__ == "__main__":
    main()