import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from veyra.features.preprocessing import build_preprocessor
from veyra.training.load_training_data import (
    create_feature_target_split,
    inspect_features,
    load_split_data,
)

def main():
    # Load train and validation once
    train_df = load_split_data("train")
    validation_df = load_split_data("validation")

    X_train, y_train = create_feature_target_split(train_df)
    X_validation, y_validation = create_feature_target_split(validation_df)

    numeric_columns, categorical_columns = inspect_features(X_train)

    experiments = [
    {
        "name": "depth_3",
        "max_depth": 3,
        "n_estimators": 300,
        "learning_rate": 0.05,
    },
    {
        "name": "depth_4",
        "max_depth": 4,
        "n_estimators": 300,
        "learning_rate": 0.05,
    },
    {
        "name": "depth_5",
        "max_depth": 5,
        "n_estimators": 300,
        "learning_rate": 0.05,
    },
    {
        "name": "depth_4_400",
        "max_depth": 4,
        "n_estimators": 400,
        "learning_rate": 0.05,
    },
    ]

    results = []

    for experiment in experiments:
        print(f"\nTraining: {experiment['name']}")

        preprocessor = build_preprocessor(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

        model = XGBClassifier(
            n_estimators=experiment["n_estimators"],
            max_depth=experiment["max_depth"],
            learning_rate=experiment["learning_rate"],
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

        pipeline.fit(X_train, y_train)

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

        results.append(
            {
                "experiment": experiment["name"],
                "max_depth": experiment["max_depth"],
                "n_estimators": experiment["n_estimators"],
                "learning_rate": experiment["learning_rate"],
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }
        )

        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
    )

    print("\nXGBoost tuning results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()