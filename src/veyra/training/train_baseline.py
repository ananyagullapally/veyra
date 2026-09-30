from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from veyra.training.load_training_data import (
    create_feature_target_split,
    inspect_features,
    load_split_data,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from veyra.features.preprocessing import build_preprocessor
from veyra.training.load_training_data import (
    create_feature_target_split,
    inspect_features,
    load_training_data,
)

def build_baseline_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build the Logistic Regression baseline pipeline."""

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline

def main():
    # Load training data
    train_df = load_split_data("train")
    X_train, y_train = create_feature_target_split(train_df)

    # Identify feature types from TRAINING data only
    numeric_columns, categorical_columns = inspect_features(X_train)

    # Build preprocessing + Logistic Regression pipeline
    pipeline = build_baseline_model(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    print("\nTraining Logistic Regression baseline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Load untouched validation period
    validation_df = load_split_data("validation")
    X_validation, y_validation = create_feature_target_split(validation_df)

    # Generate fraud scores
    validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]

    print("\nValidation predictions created:")
    print(f"Applications: {len(validation_probabilities):,}")
    print(f"Minimum score: {validation_probabilities.min():.4f}")
    print(f"Maximum score: {validation_probabilities.max():.4f}")
    print(f"Mean score:    {validation_probabilities.mean():.4f}")

    pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    print("\nValidation metrics:")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    thresholds = [0.30, 0.50, 0.70]

    print("\nThreshold analysis:")

    for threshold in thresholds:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
        ).ravel()

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")

if __name__ == "__main__":
    main()