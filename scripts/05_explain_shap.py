import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    req = {"source_id", "label", "text"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    feature_cols = [c for c in df.columns if c not in {"source_id", "label", "prompt_type", "text"}]
    X = df[feature_cols].fillna(0.0)
    y = (df["label"] == "llm").astype(int)
    groups = df["source_id"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_seed)
    _, test_idx = next(gss.split(X, y, groups=groups))
    X_test = X.iloc[test_idx]

    result_dir = Path(args.results_dir)
    model = joblib.load(result_dir / "best_model.joblib")

    sample = X_test.sample(n=min(args.sample_size, len(X_test)), random_state=args.random_seed)

    if hasattr(model, "named_steps"):
        # Pipeline case (LogisticRegression + StandardScaler)
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        sample_scaled = scaler.transform(sample)
        explainer = shap.LinearExplainer(clf, sample_scaled)
        shap_values = explainer(sample_scaled)
        sv = pd.DataFrame(shap_values.values, columns=feature_cols)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        arr = np.array(shap_values)

        # Handle common SHAP return shapes:
        # - list[class] of (n_samples, n_features)
        # - (n_samples, n_features)
        # - (n_samples, n_features, n_classes)
        # - (n_classes, n_samples, n_features)
        if isinstance(shap_values, list):
            arr = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        elif arr.ndim == 3 and arr.shape[2] == 2:
            arr = arr[:, :, 1]
        elif arr.ndim == 3 and arr.shape[0] == 2:
            arr = arr[1, :, :]
        elif arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[0, :, :]

        if arr.ndim != 2:
            raise ValueError(f"Unexpected SHAP values shape after normalization: {arr.shape}")

        sv = pd.DataFrame(arr, columns=feature_cols)

    mean_abs = sv.abs().mean().sort_values(ascending=False).reset_index()
    mean_abs.columns = ["feature", "mean_abs_shap"]
    mean_abs.to_csv(result_dir / "shap_global_importance.csv", index=False)

    local = sv.copy()
    local.insert(0, "row_index", sample.index)
    local.to_csv(result_dir / "shap_local_values.csv", index=False)

    print(f"Saved SHAP outputs in {result_dir}")


if __name__ == "__main__":
    main()
