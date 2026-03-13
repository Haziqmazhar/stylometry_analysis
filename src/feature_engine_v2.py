import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.stylometry import extract_stylometric_features, tokenize_words

LEXICAL_BASE = {
    "word_count",
    "type_count",
    "avg_word_len",
    "type_token_ratio",
    "hapax_ratio",
}

SYNTACTIC_BASE_PREFIXES = ("sentence_", "avg_sentence_", "std_sentence_", "fw_", "hedge_", "citation_", "et_al_", "punct_")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _char_ngrams(text: str, n: int) -> Counter:
    clean = re.sub(r"\s+", " ", text.lower().strip())
    grams = [clean[i : i + n] for i in range(max(len(clean) - n + 1, 0))]
    return Counter(grams)


def _writeprints_proxy(text: str) -> Dict[str, float]:
    # Writeprints-like proxy features for Python-only environments.
    words = tokenize_words(text)
    wc = len(words)
    chars = [c for c in text if not c.isspace()]
    cc = len(chars)
    upper = sum(1 for c in text if c.isupper())
    digits = sum(1 for c in text if c.isdigit())

    tri = _char_ngrams(text, 3)
    top_tri = tri.most_common(10)

    out = {
        "wp_char_per_word": _safe_div(cc, wc),
        "wp_upper_ratio": _safe_div(upper, max(len(text), 1)),
        "wp_digit_ratio": _safe_div(digits, max(len(text), 1)),
        "wp_punct_ratio": _safe_div(sum(1 for c in text if re.match(r"[^\w\s]", c)), max(len(text), 1)),
    }
    for i, (_, cnt) in enumerate(top_tri):
        out[f"wp_top_tri_{i+1}_freq"] = _safe_div(cnt, max(sum(tri.values()), 1))
    return out


def _stylometrix_proxy(text: str) -> Dict[str, float]:
    # Fallback proxy when Stylometrix package/model is unavailable.
    words = tokenize_words(text)
    wc = len(words)
    counts = Counter(words)
    long_words = sum(1 for w in words if len(w) >= 7)
    modal_verbs = sum(counts.get(w, 0) for w in ["can", "could", "may", "might", "must", "should", "would"])
    pronouns = sum(counts.get(w, 0) for w in ["i", "we", "you", "he", "she", "they", "it"])
    return {
        "sm_long_words_per100w": _safe_div(long_words * 100.0, wc),
        "sm_modal_per100w": _safe_div(modal_verbs * 100.0, wc),
        "sm_pronoun_per100w": _safe_div(pronouns * 100.0, wc),
    }


def _stanza_proxy(text: str) -> Dict[str, float]:
    # Regex proxy for POS-style distribution if stanza is unavailable.
    words = tokenize_words(text)
    wc = len(words)
    ing = sum(1 for w in words if w.endswith("ing"))
    ed = sum(1 for w in words if w.endswith("ed"))
    adv = sum(1 for w in words if w.endswith("ly"))
    tion = sum(1 for w in words if w.endswith("tion") or w.endswith("sion"))
    return {
        "st_ing_per100w": _safe_div(ing * 100.0, wc),
        "st_ed_per100w": _safe_div(ed * 100.0, wc),
        "st_adv_per100w": _safe_div(adv * 100.0, wc),
        "st_nominal_per100w": _safe_div(tion * 100.0, wc),
    }


def extract_all_features_for_text(text: str) -> Dict[str, float]:
    base = extract_stylometric_features(text)
    wp = _writeprints_proxy(text)
    sm = _stylometrix_proxy(text)
    st = _stanza_proxy(text)
    out = {}
    out.update(base)
    out.update(wp)
    out.update(sm)
    out.update(st)
    return out


def build_feature_table(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    features = df[text_col].astype(str).apply(extract_all_features_for_text).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def feature_columns_for_set(columns: Iterable[str], feature_set: str) -> List[str]:
    cols = list(columns)
    if feature_set == "lexical":
        return [c for c in cols if c in LEXICAL_BASE]
    if feature_set == "syntactic":
        return [c for c in cols if c.startswith(SYNTACTIC_BASE_PREFIXES)]
    if feature_set == "combined":
        return [c for c in cols if c in LEXICAL_BASE or c.startswith(SYNTACTIC_BASE_PREFIXES)]
    if feature_set == "writeprints":
        return [c for c in cols if c.startswith("wp_")]
    if feature_set == "stylometrix":
        return [c for c in cols if c.startswith("sm_")]
    if feature_set == "stanza":
        return [c for c in cols if c.startswith("st_")]
    return []


def ensure_nonempty_feature_set(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    kept = [c for c in feature_cols if c in df.columns]
    if not kept:
        return df, []
    out = df.copy()
    out[kept] = out[kept].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, kept
