import argparse
from pathlib import Path

import pandas as pd
try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False


ABSTRACT_CANDIDATES = ["Abstract", "abstract", "Description"]
TITLE_CANDIDATES = ["Title", "title", "Document Title"]
YEAR_CANDIDATES = ["Year", "year", "Publication Year"]
DOI_CANDIDATES = ["DOI", "doi"]
ID_CANDIDATES = ["EID", "eid", "Source ID", "source_id"]


def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def is_english(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if not HAS_LANGDETECT:
        # Fallback heuristic: keep text when language detector is unavailable.
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    abs_col = find_column(df, ABSTRACT_CANDIDATES)
    title_col = find_column(df, TITLE_CANDIDATES)
    year_col = find_column(df, YEAR_CANDIDATES)
    doi_col = find_column(df, DOI_CANDIDATES)
    id_col = find_column(df, ID_CANDIDATES)

    if not abs_col:
        raise ValueError("No abstract-like column found. Expected one of: " + ", ".join(ABSTRACT_CANDIDATES))

    work = pd.DataFrame()
    work["abstract"] = df[abs_col].astype(str).str.strip()
    work = work[work["abstract"].str.len() > 0]
    work = work[work["abstract"].apply(is_english)]

    if title_col:
        work["title"] = df.loc[work.index, title_col].astype(str).fillna("")
    else:
        work["title"] = ""

    if year_col:
        work["year"] = df.loc[work.index, year_col]
    else:
        work["year"] = None

    if doi_col:
        work["doi"] = df.loc[work.index, doi_col].astype(str).fillna("")
    else:
        work["doi"] = ""

    if id_col:
        work["source_id"] = df.loc[work.index, id_col].astype(str)
    else:
        work["source_id"] = [f"paper_{i:06d}" for i in range(len(work))]

    work = work.drop_duplicates(subset=["abstract"]).drop_duplicates(subset=["source_id"])
    work = work.head(args.limit).copy()

    cols = ["source_id", "title", "abstract", "year", "doi"]
    work = work[cols]
    work.to_csv(out_path, index=False)

    print(f"Saved {len(work)} rows to {out_path}")


if __name__ == "__main__":
    main()
