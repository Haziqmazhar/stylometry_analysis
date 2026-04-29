import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.feature_engine_v2 import (
    build_feature_table,
    ensure_nonempty_feature_set,
    feature_columns_for_set,
)


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def tokenize_for_overlap(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def _ngram_counts(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def rouge_n_f1(reference: str, candidate: str, n: int = 1) -> float:
    ref_counts = _ngram_counts(tokenize_for_overlap(reference), n)
    cand_counts = _ngram_counts(tokenize_for_overlap(candidate), n)
    if not ref_counts or not cand_counts:
        return 0.0
    overlap = 0
    for gram, count in cand_counts.items():
        overlap += min(count, ref_counts.get(gram, 0))
    ref_total = sum(ref_counts.values())
    cand_total = sum(cand_counts.values())
    if ref_total == 0 or cand_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / cand_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = tokenize_for_overlap(reference)
    cand_tokens = tokenize_for_overlap(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, cand_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def meteor_score_simple(reference: str, candidate: str) -> float:
    ref_tokens = tokenize_for_overlap(reference)
    cand_tokens = tokenize_for_overlap(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0

    ref_positions: Dict[str, List[int]] = {}
    for idx, tok in enumerate(ref_tokens):
        ref_positions.setdefault(tok, []).append(idx)

    matches = 0
    chunks = 0
    last_pos = -1
    used_ref_positions = set()

    for tok in cand_tokens:
        positions = ref_positions.get(tok, [])
        chosen = None
        for pos in positions:
            if pos not in used_ref_positions and pos > last_pos:
                chosen = pos
                break
        if chosen is None:
            for pos in positions:
                if pos not in used_ref_positions:
                    chosen = pos
                    break
        if chosen is None:
            continue
        used_ref_positions.add(chosen)
        matches += 1
        if last_pos == -1 or chosen != last_pos + 1:
            chunks += 1
        last_pos = chosen

    if matches == 0:
        return 0.0

    precision = matches / len(cand_tokens)
    recall = matches / len(ref_tokens)
    if precision == 0 or recall == 0:
        return 0.0

    f_mean = (10 * precision * recall) / (recall + 9 * precision)
    penalty = 0.5 * ((chunks / matches) ** 3) if matches > 0 else 0.0
    return f_mean * (1 - penalty)


def evaluate_semantic_overlap(cfg: Dict) -> pd.DataFrame:
    human = pd.read_csv(cfg["human_input"])
    llm = pd.read_csv(cfg["llm_input"])
    source_col = cfg.get("source_column", "source_id")
    htext = cfg.get("human_text_column", "abstract")
    ltext = cfg.get("llm_text_column", "generated_abstract")
    prompt_type = cfg.get("prompt_type", "factual")
    require_columns(human, [source_col, htext], str(cfg["human_input"]))
    require_columns(llm, [source_col, "model_name", "prompt_type", ltext], str(cfg["llm_input"]))

    llm = llm[llm["prompt_type"].astype(str).str.lower() == prompt_type.lower()].copy()
    keep_models = cfg.get("models", []) or sorted(llm["model_name"].dropna().astype(str).unique().tolist())
    llm = llm[llm["model_name"].astype(str).isin(keep_models)].copy()

    human = human[[source_col, htext]].dropna().copy()
    human = human.rename(columns={source_col: "source_id", htext: "reference_text"})
    human["source_id"] = human["source_id"].astype(str)
    human = human.drop_duplicates(subset=["source_id"])

    llm = llm[[source_col, "model_name", ltext, "prompt_type"]].dropna().copy()
    llm = llm.rename(columns={source_col: "source_id", ltext: "candidate_text"})
    llm["source_id"] = llm["source_id"].astype(str)
    llm = llm.drop_duplicates(subset=["source_id", "model_name"], keep="last")

    merged = human.merge(llm, on="source_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["rouge1_f1"] = merged.apply(
        lambda r: rouge_n_f1(r["reference_text"], r["candidate_text"], n=1), axis=1
    )
    merged["rouge2_f1"] = merged.apply(
        lambda r: rouge_n_f1(r["reference_text"], r["candidate_text"], n=2), axis=1
    )
    merged["rougeL_f1"] = merged.apply(lambda r: rouge_l_f1(r["reference_text"], r["candidate_text"]), axis=1)
    merged["meteor"] = merged.apply(lambda r: meteor_score_simple(r["reference_text"], r["candidate_text"]), axis=1)
    merged["reference_tokens"] = merged["reference_text"].map(lambda t: len(tokenize_for_overlap(t)))
    merged["candidate_tokens"] = merged["candidate_text"].map(lambda t: len(tokenize_for_overlap(t)))
    return merged


def summarize_semantic_overlap(pair_scores: pd.DataFrame) -> pd.DataFrame:
    if pair_scores.empty:
        return pd.DataFrame()
    metrics = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "meteor", "reference_tokens", "candidate_tokens"]
    grouped = pair_scores.groupby("model_name")[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["model_name"] + [f"{metric}_{stat}" for metric, stat in grouped.columns.tolist()[1:]]
    return grouped.sort_values("model_name").reset_index(drop=True)


def build_master_dataset(cfg: Dict) -> pd.DataFrame:
    human = pd.read_csv(cfg["human_input"])
    llm = pd.read_csv(cfg["llm_input"])
    source_col = cfg.get("source_column", "source_id")
    htext = cfg.get("human_text_column", "abstract")
    ltext = cfg.get("llm_text_column", "generated_abstract")
    prompt_type = cfg.get("prompt_type", "factual")
    require_columns(human, [source_col, htext], str(cfg["human_input"]))
    require_columns(llm, [source_col, "model_name", "prompt_type", ltext], str(cfg["llm_input"]))

    human_rows = human[[source_col, htext]].copy()
    human_rows = human_rows.rename(columns={htext: "text", source_col: "source_id"})
    human_rows["author_label"] = "human"
    human_rows["model_name"] = "human"

    llm = llm[llm["prompt_type"].astype(str).str.lower() == prompt_type.lower()].copy()
    keep_models = cfg.get("models", []) or sorted(llm["model_name"].dropna().astype(str).unique().tolist())
    llm = llm[llm["model_name"].astype(str).isin(keep_models)].copy()
    llm_rows = llm[[source_col, "model_name", ltext]].copy()
    llm_rows = llm_rows.rename(columns={ltext: "text", source_col: "source_id"})
    llm_rows["author_label"] = llm_rows["model_name"].astype(str)

    common = set(human_rows["source_id"].astype(str))
    for m in keep_models:
        ids = set(llm_rows.loc[llm_rows["model_name"] == m, "source_id"].astype(str))
        common = common.intersection(ids)
    common_ids = sorted(common)

    human_rows = human_rows[human_rows["source_id"].astype(str).isin(common_ids)].drop_duplicates(subset=["source_id"])
    llm_rows = llm_rows[llm_rows["source_id"].astype(str).isin(common_ids)].drop_duplicates(subset=["source_id", "model_name"])

    out = pd.concat([human_rows, llm_rows], ignore_index=True)
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def _safe_roc_auc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def evaluate_binary_cv(df: pd.DataFrame, feature_cols: List[str], folds: int, seed: int) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"logreg": [], "random_forest": []}
    x = df[feature_cols].astype(float)
    y = (df["label"] == 1).astype(int).values
    groups = df["source_id"].astype(str).values

    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(x, y, groups), start=1):
        xtr, xte = x.iloc[tr], x.iloc[te]
        ytr, yte = y[tr], y[te]

        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
        )

        for name, model in [("logreg", lr), ("random_forest", rf)]:
            model.fit(xtr, ytr)
            pred = model.predict(xte)
            prob = model.predict_proba(xte)[:, 1]
            out[name].append(
                {
                    "fold": fold,
                    "accuracy": accuracy_score(yte, pred),
                    "precision": precision_score(yte, pred, zero_division=0),
                    "recall": recall_score(yte, pred, zero_division=0),
                    "f1": f1_score(yte, pred, zero_division=0),
                    "roc_auc": _safe_roc_auc_binary(yte, prob),
                    "mcc": matthews_corrcoef(yte, pred),
                    "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                }
            )
    return out


def evaluate_multiclass_cv(df: pd.DataFrame, feature_cols: List[str], folds: int, seed: int) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"logreg": [], "random_forest": []}
    x = df[feature_cols].astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["author_label"].astype(str).values)
    groups = df["source_id"].astype(str).values
    n_classes = len(le.classes_)

    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(x, y, groups), start=1):
        xtr, xte = x.iloc[tr], x.iloc[te]
        ytr, yte = y[tr], y[te]

        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
            ]
        )
        rf = RandomForestClassifier(
            n_estimators=700,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced_subsample",
        )

        for name, model in [("logreg", lr), ("random_forest", rf)]:
            model.fit(xtr, ytr)
            pred = model.predict(xte)
            row = {
                "fold": fold,
                "accuracy": accuracy_score(yte, pred),
                "precision": precision_score(yte, pred, average="macro", zero_division=0),
                "recall": recall_score(yte, pred, average="macro", zero_division=0),
                "f1": f1_score(yte, pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            }
            try:
                prob = model.predict_proba(xte)
                if prob.shape[1] == n_classes:
                    row["roc_auc"] = roc_auc_score(yte, prob, multi_class="ovr", average="macro")
                else:
                    row["roc_auc"] = float("nan")
            except Exception:
                row["roc_auc"] = float("nan")
            out[name].append(row)
    return out


def summarize(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for c in [c for c in df.columns if c != "fold"]:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    acfg = cfg["analysis"]
    out_dir = Path(acfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    master = build_master_dataset(acfg)
    if master.empty:
        raise ValueError("No source-matched records were available for analysis. Check inputs, models, and prompt_type.")
    master = build_feature_table(master, text_col="text")
    master.to_csv(out_dir / "master_with_features.csv", index=False)

    if acfg.get("run_semantic_eval", True):
        semantic_pairs = evaluate_semantic_overlap(acfg)
        if not semantic_pairs.empty:
            semantic_pairs.to_csv(out_dir / "semantic_pair_scores.csv", index=False)
            semantic_summary = summarize_semantic_overlap(semantic_pairs)
            semantic_summary.to_csv(out_dir / "summary_semantic.csv", index=False)

    meta_cols = {"source_id", "text", "author_label", "model_name"}
    all_feature_cols = [c for c in master.columns if c not in meta_cols]

    folds = int(acfg.get("cv_folds", 5))
    seed = int(acfg.get("random_seed", 42))
    feature_sets = acfg.get("feature_sets", ["lexical", "syntactic", "combined"])

    binary_summary = []
    multiclass_summary = []

    if acfg.get("run_binary", True):
        llm_models = sorted([m for m in master["model_name"].unique() if m != "human"])
        for model_name in llm_models:
            subset = master[(master["model_name"] == "human") | (master["model_name"] == model_name)].copy()
            subset["label"] = (subset["model_name"] == model_name).astype(int)
            for fs in feature_sets:
                fs_cols = feature_columns_for_set(all_feature_cols, fs)
                subset2, fs_cols = ensure_nonempty_feature_set(subset, fs_cols)
                if not fs_cols:
                    continue
                cvres = evaluate_binary_cv(subset2, fs_cols, folds, seed)
                for model_type, rows in cvres.items():
                    summary = summarize(rows)
                    summary.update(
                        {
                            "task": "binary",
                            "target_model": model_name,
                            "feature_set": fs,
                            "estimator": model_type,
                            "n_features": len(fs_cols),
                            "n_rows": len(subset2),
                        }
                    )
                    binary_summary.append(summary)
                pd.DataFrame(cvres["logreg"]).to_csv(out_dir / f"binary_{model_name}_{fs}_logreg_folds.csv", index=False)
                pd.DataFrame(cvres["random_forest"]).to_csv(
                    out_dir / f"binary_{model_name}_{fs}_random_forest_folds.csv", index=False
                )

    if acfg.get("run_multiclass", False):
        for fs in feature_sets:
            fs_cols = feature_columns_for_set(all_feature_cols, fs)
            data2, fs_cols = ensure_nonempty_feature_set(master, fs_cols)
            if not fs_cols:
                continue
            cvres = evaluate_multiclass_cv(data2, fs_cols, folds, seed)
            for model_type, rows in cvres.items():
                summary = summarize(rows)
                summary.update(
                    {
                        "task": "multiclass",
                        "target_model": "all",
                        "feature_set": fs,
                        "estimator": model_type,
                        "n_features": len(fs_cols),
                        "n_rows": len(data2),
                        "n_classes": int(data2["author_label"].nunique()),
                    }
                )
                multiclass_summary.append(summary)
            pd.DataFrame(cvres["logreg"]).to_csv(out_dir / f"multiclass_{fs}_logreg_folds.csv", index=False)
            pd.DataFrame(cvres["random_forest"]).to_csv(out_dir / f"multiclass_{fs}_random_forest_folds.csv", index=False)

    bdf = pd.DataFrame(binary_summary)
    mdf = pd.DataFrame(multiclass_summary)
    if not bdf.empty:
        bdf.to_csv(out_dir / "summary_binary.csv", index=False)
    if not mdf.empty:
        mdf.to_csv(out_dir / "summary_multiclass.csv", index=False)

    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_folds": folds,
                "random_seed": seed,
                "feature_sets": feature_sets,
                "binary_rows": len(binary_summary),
                "multiclass_rows": len(multiclass_summary),
                "semantic_eval_enabled": bool(acfg.get("run_semantic_eval", True)),
            },
            f,
            indent=2,
        )

    print(f"Saved v2 analysis outputs to {out_dir}")
    print("Multiclass means one classifier predicts among all classes simultaneously:")
    print("human + each LLM model label, not just human vs one model.")


if __name__ == "__main__":
    main()
