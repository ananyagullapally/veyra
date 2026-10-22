import mlflow
import mlflow.sklearn

from veyra.training.mlflow_setup import configure_mlflow
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
from veyra.training.threshold import select_threshold_for_recall


def main():
    # Load training data
    train_df = load_split_data("train")
    X_train, y_train = create_feature_target_split(train_df)

    numeric_columns, categorical_columns = inspect_features(X_train)

    # Build selected XGBoost candidate
    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
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

    print("\nTraining selected XGBoost candidate...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Month 6 validation
    validation_df = load_split_data("validation")
    X_validation, y_validation = create_feature_target_split(
        validation_df
    )

    probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    # Select operating threshold using validation only
    threshold_result = select_threshold_for_recall(
        y_validation,
        probabilities,
        minimum_recall=0.80,
    )

    threshold = threshold_result["threshold"]

    predictions = (
        probabilities >= threshold
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

    print("\nSelected candidate validation results:")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nConfusion matrix:")
    print(
        f"TP: {tp:,} | FP: {fp:,} | "
        f"FN: {fn:,} | TN: {tn:,}"
    )

    configure_mlflow()

    with mlflow.start_run(run_name="xgboost-candidate") as run:
        mlflow.log_params({
            "model_type": "XGBoost",
            "max_depth": 3,
            "selected_threshold": float(threshold),
        })

        mlflow.log_metrics({
            "validation_pr_auc": pr_auc,
            "validation_roc_auc": roc_auc,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
        })

        mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="fraud_pipeline",
        skops_trusted_types=[
            "numpy.dtype",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )

    print("\nModel and metrics logged to MLflow!")
    return run.info.run_id

if __name__ == "__main__":
    main()