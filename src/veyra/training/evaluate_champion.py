import mlflow.sklearn

from veyra.training.model_registry import get_champion_info
from veyra.training.evaluate_model import evaluate_model
from veyra.training.load_training_data import (
    load_split_data,
    create_feature_target_split,
)


def main():
    champion = get_champion_info()

    print(f"Champion version: {champion['version']}")
    print(f"Threshold: {champion['threshold']}")

    pipeline = mlflow.sklearn.load_model(
        champion["model_uri"]
    )

    validation_df = load_split_data("validation")

    X_validation, y_validation = create_feature_target_split(
        validation_df
    )

    metrics = evaluate_model(
        pipeline=pipeline,
        X=X_validation,
        y=y_validation,
        threshold=champion["threshold"],
    )

    print("\nChampion validation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    return metrics


if __name__ == "__main__":
    main()