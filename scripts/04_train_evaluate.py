import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    req = {"source_id", "label", "text"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    feature_cols = [
        c for c in df.columns
        if c not in {"source_id", "label", "prompt_type", "text"}
    ]
    X = df[feature_cols].fillna(0.0)
    y = (df["label"] == "llm").astype(int)
    groups = df["source_id"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=args.random_seed)),
    ])
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=args.random_seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    results = {
        "logistic_regression": compute_metrics(y_test, lr_pred, lr_prob),
        "random_forest": compute_metrics(y_test, rf_pred, rf_prob),
    }

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = lr if best_name == "logistic_regression" else rf
    best_pred = lr_pred if best_name == "logistic_regression" else rf_pred

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm = confusion_matrix(y_test, best_pred)
    pd.DataFrame(cm, index=["human", "llm"], columns=["pred_human", "pred_llm"]).to_csv(
        result_dir / "confusion_matrix_best.csv"
    )

    with (result_dir / "classification_report_best.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_test, best_pred, target_names=["human", "llm"]))

    if best_name == "logistic_regression":
        coef = best_model.named_steps["clf"].coef_[0]
        imp = pd.DataFrame({"feature": feature_cols, "importance": np.abs(coef)})
        imp = imp.sort_values("importance", ascending=False)
    else:
        imp = pd.DataFrame({"feature": feature_cols, "importance": best_model.feature_importances_})
        imp = imp.sort_values("importance", ascending=False)

    imp.to_csv(result_dir / "feature_importance_global.csv", index=False)

    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=args.random_seed)
    perm_df = pd.DataFrame(
        {"feature": feature_cols, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(result_dir / "feature_importance_permutation.csv", index=False)

    joblib.dump(best_model, result_dir / "best_model.joblib")
    pd.DataFrame({"idx": test_idx, "y_true": y_test.values, "y_pred": best_pred}).to_csv(
        result_dir / "test_predictions.csv", index=False
    )

    print(f"Saved results in {result_dir}")
    print("Best model:", best_name)
    print(json.dumps(results[best_name], indent=2))


if __name__ == "__main__":
    main()
