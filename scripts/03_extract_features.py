import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure `src` is importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stylometry import extract_stylometric_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--prompt-type", default="factual")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    human = pd.read_csv(args.human)
    llm = pd.read_csv(args.llm)

    required_human = {"source_id", "abstract"}
    required_llm = {"source_id", "generated_abstract", "prompt_type"}
    miss_h = required_human - set(human.columns)
    miss_l = required_llm - set(llm.columns)
    if miss_h:
        raise ValueError(f"Missing human columns: {sorted(miss_h)}")
    if miss_l:
        raise ValueError(f"Missing llm columns: {sorted(miss_l)}")

    llm = llm[llm["prompt_type"].astype(str).str.lower() == args.prompt_type.lower()].copy()

    human_rows = human[["source_id", "abstract"]].copy()
    human_rows["label"] = "human"
    human_rows["prompt_type"] = "human"
    human_rows = human_rows.rename(columns={"abstract": "text"})

    llm_rows = llm[["source_id", "generated_abstract", "prompt_type"]].copy()
    llm_rows["label"] = "llm"
    llm_rows = llm_rows.rename(columns={"generated_abstract": "text"})

    keep_sources = set(human_rows["source_id"]).intersection(set(llm_rows["source_id"]))
    human_rows = human_rows[human_rows["source_id"].isin(keep_sources)].copy()
    llm_rows = llm_rows[llm_rows["source_id"].isin(keep_sources)].copy()

    # Keep one human and one llm instance per source_id for a balanced paired setup.
    human_rows = human_rows.drop_duplicates(subset=["source_id"])
    llm_rows = llm_rows.drop_duplicates(subset=["source_id"])

    n = min(len(human_rows), len(llm_rows))
    human_rows = human_rows.head(n)
    llm_rows = llm_rows.set_index("source_id").loc[human_rows["source_id"]].reset_index()

    combined = pd.concat([human_rows, llm_rows], ignore_index=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0].copy()

    feats = combined["text"].apply(extract_stylometric_features).apply(pd.Series)
    out = pd.concat([combined.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved features: {len(out)} rows, {len(feats.columns)} features -> {out_path}")


if __name__ == "__main__":
    main()
