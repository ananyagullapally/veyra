from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from veyra.features.preprocessing import build_preprocessor
from veyra.training.load_training_data import (
    create_feature_target_split,
    inspect_features,
    load_split_data,
)


def build_xgboost_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    """Build the XGBoost fraud detection pipeline."""

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline

def main():
    # Load training period
    train_df = load_split_data("train")
    X_train, y_train = create_feature_target_split(train_df)

    numeric_columns, categorical_columns = inspect_features(X_train)

    # Calculate imbalance from training labels
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("\nClass imbalance:")
    print(f"Non-fraud:        {negative_count:,}")
    print(f"Fraud:            {positive_count:,}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    pipeline = build_xgboost_model(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scale_pos_weight=scale_pos_weight,
    )

    print("\nTraining XGBoost baseline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on Month 6 only
    validation_df = load_split_data("validation")
    X_validation, y_validation = create_feature_target_split(validation_df)

    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    print("\nXGBoost validation metrics:")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    thresholds = [0.30, 0.50, 0.70]

    print("\nXGBoost threshold analysis:")

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
        print(
            f"TP: {tp:,} | FP: {fp:,} | "
            f"FN: {fn:,} | TN: {tn:,}"
        )

if __name__ == "__main__":
    main()
