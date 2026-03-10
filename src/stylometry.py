import math
import re
from collections import Counter
from typing import Dict, List

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"[.!?]+")

FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "while", "because", "as",
    "of", "in", "on", "for", "to", "from", "by", "with", "without", "about",
    "at", "into", "through", "between", "among", "is", "are", "was", "were",
    "be", "been", "being", "that", "this", "these", "those", "it", "its",
    "their", "there", "which", "who", "whom", "whose", "not", "no", "can",
    "could", "may", "might", "must", "should", "would",
]

HEDGES = [
    "may", "might", "could", "possibly", "likely", "suggests", "indicates",
    "approximately", "potentially", "appears", "seems",
]

PUNCT_CHARS = [",", ";", ":", "-", "?", "!", "(", ")", "\""]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    chunks = [c.strip() for c in SENT_SPLIT_RE.split(text) if c.strip()]
    return chunks if chunks else [text.strip()] if text.strip() else []


def punctuation_rates(text: str, word_count: int) -> Dict[str, float]:
    rates = {}
    for ch in PUNCT_CHARS:
        count = text.count(ch)
        rates[f"punct_{ord(ch)}_per100w"] = safe_div(count * 100.0, word_count)
    return rates


def lexical_features(words: List[str]) -> Dict[str, float]:
    wc = len(words)
    counts = Counter(words)
    types = len(counts)
    hapax = sum(1 for _, c in counts.items() if c == 1)

    avg_word_len = safe_div(sum(len(w) for w in words), wc)
    ttr = safe_div(types, wc)
    hapax_ratio = safe_div(hapax, wc)

    return {
        "word_count": float(wc),
        "type_count": float(types),
        "avg_word_len": avg_word_len,
        "type_token_ratio": ttr,
        "hapax_ratio": hapax_ratio,
    }


def sentence_features(sentences: List[str], words: List[str]) -> Dict[str, float]:
    sent_word_lens = []
    for s in sentences:
        sent_word_lens.append(len(tokenize_words(s)))

    avg_len = safe_div(sum(sent_word_lens), len(sent_word_lens))
    if sent_word_lens:
        mean = avg_len
        var = safe_div(sum((x - mean) ** 2 for x in sent_word_lens), len(sent_word_lens))
        std = math.sqrt(var)
    else:
        std = 0.0

    return {
        "sentence_count": float(len(sentences)),
        "avg_sentence_len_words": avg_len,
        "std_sentence_len_words": std,
    }


def function_word_features(words: List[str]) -> Dict[str, float]:
    wc = len(words)
    counts = Counter(words)
    feats = {}
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}_per100w"] = safe_div(counts.get(fw, 0) * 100.0, wc)
    return feats


def hedge_features(words: List[str]) -> Dict[str, float]:
    wc = len(words)
    counts = Counter(words)
    hedge_count = sum(counts.get(h, 0) for h in HEDGES)
    return {"hedge_per100w": safe_div(hedge_count * 100.0, wc)}


def citation_features(text: str, words: List[str]) -> Dict[str, float]:
    wc = len(words)
    paren_year = len(re.findall(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)", text))
    et_al = len(re.findall(r"\bet al\.\b", text.lower()))
    return {
        "citation_year_per100w": safe_div(paren_year * 100.0, wc),
        "et_al_per100w": safe_div(et_al * 100.0, wc),
    }


def extract_stylometric_features(text: str) -> Dict[str, float]:
    text = text or ""
    words = tokenize_words(text)
    sentences = split_sentences(text)

    features = {}
    features.update(lexical_features(words))
    features.update(sentence_features(sentences, words))
    features.update(punctuation_rates(text, int(features["word_count"])))
    features.update(function_word_features(words))
    features.update(hedge_features(words))
    features.update(citation_features(text, words))
    return features
