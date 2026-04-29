import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURE_SETS = ["lexical", "syntactic", "combined", "writeprints", "stylometrix", "stanza"]
FEATURE_LABELS = {
    "lexical": "Lexical",
    "syntactic": "Syntactic",
    "combined": "Combined",
    "writeprints": "Writeprints",
    "stylometrix": "Stylometrix",
    "stanza": "Stanza",
}
ESTIMATOR_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
}
METRICS = ["accuracy", "f1", "roc_auc", "mcc"]


def parse_binary_fold_name(path: Path) -> Dict[str, str] | None:
    name = path.name
    if not name.startswith("binary_") or not name.endswith("_folds.csv"):
        return None

    stem = name[len("binary_") : -len("_folds.csv")]
    estimator = None
    for suffix in ["_logreg", "_random_forest"]:
        if stem.endswith(suffix):
            estimator = suffix[1:]
            stem = stem[: -len(suffix)]
            break
    if estimator is None:
        return None

    feature_set = None
    target_model = None
    for fs in FEATURE_SETS:
        token = f"_{fs}"
        if stem.endswith(token):
            feature_set = fs
            target_model = stem[: -len(token)]
            break
    if feature_set is None or not target_model:
        return None

    return {
        "target_model": target_model,
        "feature_set": feature_set,
        "estimator": estimator,
    }


def load_binary_fold_results(results_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in sorted(results_dir.glob("binary_*_folds.csv")):
        meta = parse_binary_fold_name(path)
        if meta is None:
            continue
        df = pd.read_csv(path)
        for key, value in meta.items():
            df[key] = value
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["estimator_label"] = out["estimator"].map(ESTIMATOR_LABELS).fillna(out["estimator"])
    out["feature_label"] = out["feature_set"].map(FEATURE_LABELS).fillna(out["feature_set"])
    out["feature_set"] = pd.Categorical(out["feature_set"], categories=FEATURE_SETS, ordered=True)
    return out


def summarize_results(folds_df: pd.DataFrame) -> pd.DataFrame:
    if folds_df.empty:
        return pd.DataFrame()
    grouped = (
        folds_df.groupby(
            ["target_model", "feature_set", "feature_label", "estimator", "estimator_label"], observed=True
        )[METRICS]
        .mean()
        .reset_index()
    )
    return grouped.sort_values(["target_model", "feature_set", "estimator"])


def plot_metric_bars(summary_df: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["accuracy", "f1", "roc_auc"]:
        if metric not in summary_df.columns:
            continue
        targets = summary_df["target_model"].nunique()
        plt.figure(figsize=(10.5, 5.5))
        ax = sns.barplot(
            data=summary_df,
            y="feature_label",
            x=metric,
            hue="estimator_label",
            palette="Set2",
            orient="h",
        )
        ax.set_title(f"Mean {metric.upper()} by Feature Set")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Feature set")
        ax.set_xlim(0, 1)
        if targets == 1:
            target_label = summary_df["target_model"].iloc[0]
            ax.set_title(f"Mean {metric.upper()} by Feature Set ({target_label})")
        else:
            ax.set_title(f"Mean {metric.upper()} by Feature Set Across Binary Tasks")
        ax.legend(title="Estimator", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_by_feature_set.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_binary_best_f1_by_model(summary_df: pd.DataFrame, output_dir: Path) -> None:
    best = (
        summary_df.sort_values("f1", ascending=False)
        .groupby(["target_model", "feature_set", "feature_label"], observed=True)
        .head(1)
        .copy()
    )
    best["target_model"] = pd.Categorical(
        best["target_model"],
        categories=sorted(best["target_model"].astype(str).unique().tolist()),
        ordered=True,
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=best.sort_values(["feature_set", "target_model"]),
        y="feature_label",
        x="f1",
        hue="target_model",
        palette="Set2",
        orient="h",
    )
    ax.set_title("Best Binary F1 by Feature Set and Model")
    ax.set_xlabel("Mean F1")
    ax.set_ylabel("Feature set")
    ax.set_xlim(0, 1)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_dir / "binary_best_f1_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_metric_bars(results_dir: Path, output_dir: Path) -> None:
    summary_path = results_dir / "summary_multiclass.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        return
    df["feature_label"] = df["feature_set"].map(FEATURE_LABELS).fillna(df["feature_set"])
    df["estimator_label"] = df["estimator"].map(ESTIMATOR_LABELS).fillna(df["estimator"])
    for metric_col, label in [("f1_mean", "Mean Macro-F1"), ("accuracy_mean", "Mean Accuracy")]:
        plt.figure(figsize=(10.5, 5.5))
        ax = sns.barplot(
            data=df.sort_values(["feature_set", "estimator"]),
            y="feature_label",
            x=metric_col,
            hue="estimator_label",
            palette="Set2",
            orient="h",
        )
        ax.set_title(f"Multiclass {label} by Feature Set")
        ax.set_xlabel(label)
        ax.set_ylabel("Feature set")
        ax.set_xlim(0, 1)
        ax.legend(title="Estimator", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(output_dir / f"multiclass_{metric_col}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_fold_distributions(folds_df: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["accuracy", "f1"]:
        plt.figure(figsize=(10.5, 5.5))
        ax = sns.boxplot(
            data=folds_df,
            y="feature_label",
            x=metric,
            hue="estimator_label",
            palette="Set2",
            orient="h",
        )
        sns.stripplot(
            data=folds_df,
            y="feature_label",
            x=metric,
            hue="estimator_label",
            dodge=True,
            alpha=0.6,
            palette="dark:black",
            orient="h",
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[: len(ESTIMATOR_LABELS)],
            labels[: len(ESTIMATOR_LABELS)],
            title="Estimator",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )
        ax.set_title(f"Fold-Level {metric.upper()} Distribution")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Feature set")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_fold_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_class_balance(results_dir: Path, output_dir: Path) -> None:
    master_path = results_dir / "master_with_features.csv"
    if not master_path.exists():
        return
    master = pd.read_csv(master_path)
    if "author_label" not in master.columns:
        return
    counts = master["author_label"].astype(str).value_counts().reset_index()
    counts.columns = ["author_label", "count"]
    colors = sns.color_palette("Set2", n_colors=len(counts))
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts["count"],
        labels=counts["author_label"],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    plt.title("Class Balance in master_with_features.csv")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_coverage(summary_df: pd.DataFrame, output_dir: Path) -> None:
    coverage_rows = []
    targets = sorted(summary_df["target_model"].astype(str).unique().tolist())
    estimators = sorted(summary_df["estimator_label"].astype(str).unique().tolist())
    for target in targets:
        for feature_set in FEATURE_SETS:
            label = FEATURE_LABELS[feature_set]
            for estimator in estimators:
                has_result = not summary_df[
                    (summary_df["target_model"] == target)
                    & (summary_df["feature_set"] == feature_set)
                    & (summary_df["estimator_label"] == estimator)
                ].empty
                coverage_rows.append(
                    {
                        "target_model": target,
                        "feature_label": label,
                        "estimator_label": estimator,
                        "available": 1 if has_result else 0,
                    }
                )
    coverage = pd.DataFrame(coverage_rows)
    if coverage.empty:
        return
    for target, target_df in coverage.groupby("target_model", observed=True):
        heatmap_df = target_df.pivot(index="feature_label", columns="estimator_label", values="available")
        plt.figure(figsize=(6, 4.8))
        ax = sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".0f",
            cmap=sns.color_palette(["#e5e7eb", "#0f766e"], as_cmap=True),
            cbar=False,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"Saved Result Coverage ({target})")
        ax.set_xlabel("Estimator")
        ax.set_ylabel("Feature set")
        plt.tight_layout()
        plt.savefig(output_dir / f"{target}_feature_coverage.png", dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="reports/results/v2")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_df = load_binary_fold_results(results_dir)
    if folds_df.empty:
        raise ValueError(f"No binary fold CSV files found in {results_dir}")

    summary_df = summarize_results(folds_df)
    summary_df.to_csv(output_dir / "binary_metric_summary.csv", index=False)

    plot_metric_bars(summary_df, output_dir)
    plot_fold_distributions(folds_df, output_dir)
    plot_class_balance(results_dir, output_dir)
    plot_binary_best_f1_by_model(summary_df, output_dir)
    plot_multiclass_metric_bars(results_dir, output_dir)

    print(f"Saved visualisations to {output_dir}")


if __name__ == "__main__":
    main()
