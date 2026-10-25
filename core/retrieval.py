"""
core/retrieval.py
Implements Good (TF-IDF), Better (BM25), and Best (Hybrid Multi-Signal Ensemble)
retrieval pipelines with date-aware boosting and quality tiering.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.query_understanding import (
    QueryAnalysis,
    TargetDate,
    extract_date,
    normalize_text,
    tokenize,
)
from web.sources import get_source_quality


def build_tfidf_index(records: List[Dict]) -> Tuple[TfidfVectorizer, Any]:
    """Build word-level TF-IDF vectorizer (unigrams + bigrams) and matrix."""
    corpus = [
        f"{r.get('title', '')}. {r.get('content', '')}" if r.get("title") else r.get("content", "")
        for r in records
    ]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def build_bm25_index(records: List[Dict]) -> Tuple[List[List[str]], Dict[str, float], float]:
    """Build BM25 index components: tokenized corpus, Robertson-Spärck Jones IDF, avgdl."""
    tokenized = [
        tokenize(f"{r.get('title', '')} {r.get('content', '')}")
        for r in records
    ]
    n_docs = len(tokenized)
    avgdl = (sum(len(d) for d in tokenized) / n_docs) if n_docs > 0 else 0.0

    df: Dict[str, int] = {}
    for doc in tokenized:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    idf: Dict[str, float] = {
        term: float(np.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)))
        for term, freq in df.items()
    }
    return tokenized, idf, avgdl


def calculate_bm25_scores(
    query: str,
    records: List[Dict],
    k1: float = 1.5,
    b: float = 0.75
) -> np.ndarray:
    """Calculate raw Okapi BM25 scores for all records."""
    tokenized, idf, avgdl = build_bm25_index(records)
    q_terms = tokenize(query)
    scores = np.zeros(len(records), dtype=float)

    for i, doc in enumerate(tokenized):
        if not doc:
            continue
        counts: Dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        dl = len(doc)
        for term in q_terms:
            if term not in counts:
                continue
            tf = counts[term]
            denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            scores[i] += idf.get(term, 0.0) * (tf * (k1 + 1.0)) / denom

    return scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Safely min-max or max-normalize score array to range [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    max_val = np.max(scores)
    min_val = np.min(scores)
    if max_val <= 0 or (max_val == min_val):
        return np.zeros_like(scores) if max_val <= 0 else np.ones_like(scores) * 0.5
    return (scores - min_val) / (max_val - min_val)


def calculate_date_scores(records: List[Dict], target_date: Optional[TargetDate]) -> np.ndarray:
    """
    Date-Aware relevance:
    Strongly boosts exact date matches.
    Penalizes conflicting dates when an exact target date was requested.
    """
    n = len(records)
    if not target_date or n == 0:
        return np.zeros(n, dtype=float)

    scores = np.zeros(n, dtype=float)
    target_iso = target_date.iso_date
    target_str = target_date.display_str.lower()
    target_month = target_date.month_name.lower()
    target_day = str(target_date.day)
    target_year = str(target_date.year)

    for i, r in enumerate(records):
        content = r.get("content", "").lower()
        title = r.get("title", "").lower()
        pub_date = r.get("pub_date", "") or r.get("detected_date", "")

        # Check explicit metadata pub_date or detected_date
        if pub_date:
            if target_iso in pub_date or target_str in pub_date.lower():
                scores[i] = 1.0
                continue

        # Check in content and title
        full_text = f"{title} {content}"
        # Direct exact date match
        if target_str in full_text or target_iso in full_text:
            scores[i] = 1.0
            continue

        # Month and Day and Year present in proximity
        if target_month in full_text and target_year in full_text:
            if re.search(rf"\b{target_day}(?:st|nd|rd|th)?\s+{target_month}\b|\b{target_month}\s+{target_day}\b", full_text):
                scores[i] = 0.95
                continue

        # Year match only
        if target_year in full_text:
            scores[i] = 0.20

        # Conflicting date penalty: if query asks for 25 Oct 2026, and record has "19 Oct 2026" or "24 Oct 2026"
        other_date = extract_date(full_text)
        if other_date and other_date.iso_date != target_iso:
            scores[i] = -0.30

    return scores


def calculate_entity_scores(records: List[Dict], entities: List[str]) -> np.ndarray:
    """Calculates entity overlap ratio between query entities and record text."""
    n = len(records)
    if not entities or n == 0:
        return np.zeros(n, dtype=float)

    scores = np.zeros(n, dtype=float)
    ent_lower = [e.lower() for e in entities]

    for i, r in enumerate(records):
        text = f"{r.get('title', '')} {r.get('content', '')}".lower()
        matches = sum(1 for e in ent_lower if e in text)
        scores[i] = matches / len(ent_lower)

    return scores


def calculate_source_scores(records: List[Dict]) -> np.ndarray:
    """Retrieves source quality scores for each record."""
    scores = np.zeros(len(records), dtype=float)
    for i, r in enumerate(records):
        # If record already has precomputed quality_score
        if "quality_score" in r:
            scores[i] = float(r["quality_score"])
        else:
            q, _, _ = get_source_quality(r.get("url", ""), r.get("publisher", ""))
            scores[i] = q
    return scores


def retrieve_candidates(
    question: str,
    records: List[Dict],
    analysis: QueryAnalysis,
    method: str = "Best — Hybrid + Reranking",
    top_k: int = 5
) -> Tuple[List[Tuple[Dict, float]], np.ndarray]:
    """
    Ranks records using selected algorithm mode.
    Returns:
      - sorted list of (record, normalized_score)
      - array of raw scores for all records
    """
    if not records:
        return [], np.array([])

    n = len(records)
    vectorizer, matrix = build_tfidf_index(records)
    qv = vectorizer.transform([question])
    tfidf_raw = cosine_similarity(qv, matrix).ravel()
    tfidf_norm = normalize_scores(tfidf_raw)

    bm25_raw = calculate_bm25_scores(question, records)
    bm25_norm = normalize_scores(bm25_raw)

    m_clean = (method or "").replace("—", "-").replace("–", "-").lower()

    if ("tf-idf" in m_clean or "good" in m_clean) and "bm25" not in m_clean and "hybrid" not in m_clean:
        final_scores = tfidf_norm

    elif ("bm25" in m_clean or "better" in m_clean) and "tf-idf" not in m_clean and "hybrid" not in m_clean:
        final_scores = bm25_norm

    elif "hybrid (tf-idf + bm25)" in m_clean or "hybrid tf-idf" in m_clean:
        final_scores = 0.55 * tfidf_norm + 0.45 * bm25_norm

    else:
        # BEST: Multi-Signal Ensemble
        # 1. Character n-grams for spelling/morphology robustness
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        char_matrix = char_vec.fit_transform([r.get("content", "") for r in records])
        char_qv = char_vec.transform([question])
        char_norm = normalize_scores(cosine_similarity(char_qv, char_matrix).ravel())

        # 2. Query keyword coverage
        kw_set = set(analysis.keywords)
        coverage = np.zeros(n, dtype=float)
        for i, r in enumerate(records):
            doc_toks = set(tokenize(r.get("content", "")))
            coverage[i] = len(kw_set & doc_toks) / max(len(kw_set), 1)

        # 3. Exact phrase match
        q_norm = normalize_text(question.lower())
        phrase_scores = np.array([
            1.0 if q_norm and q_norm in r.get("content", "").lower() else 0.0
            for r in records
        ], dtype=float)

        # 4. Title term match
        title_scores = np.zeros(n, dtype=float)
        for i, r in enumerate(records):
            t_toks = set(tokenize(r.get("title", "")))
            title_scores[i] = len(kw_set & t_toks) / max(len(kw_set), 1)

        # 5. Date-aware signal
        date_scores = calculate_date_scores(records, analysis.target_date)

        # 6. Entity overlap
        entity_scores = calculate_entity_scores(records, analysis.entities)

        # 7. Intent-specific matching
        intent_scores = np.zeros(n, dtype=float)
        if analysis.intent == "definition":
            for i, r in enumerate(records):
                if re.search(r"\b(is|are|refers to|means|stands for|defined as)\b", r.get("content", ""), re.I):
                    intent_scores[i] = 1.0
        elif analysis.intent == "sports":
            for i, r in enumerate(records):
                if re.search(r"\b(won|winner|champion|defeated|final|score|trophy|gold)\b", r.get("content", ""), re.I):
                    intent_scores[i] = 1.0
        elif analysis.intent == "daily_news_summary" or analysis.intent == "latest_news":
            for i, r in enumerate(records):
                if re.search(r"\b(announced|reported|said|launched|signed|held|killed|struck|agreed|won|began)\b", r.get("content", ""), re.I):
                    intent_scores[i] = 0.5

        # 8. Source quality tier
        source_scores = calculate_source_scores(records)

        # 9. Recency signal for latest news
        recency_scores = np.zeros(n, dtype=float)
        if analysis.is_latest or analysis.intent in ["latest_news", "daily_news_summary"]:
            for i, r in enumerate(records):
                pub = r.get("pub_date", "") or r.get("detected_date", "")
                full_l = f"{r.get('title', '')} {r.get('content', '')}".lower()
                if pub:
                    recency_scores[i] = 1.0
                # Penalize historical/ancient encyclopedic entries for latest news
                if re.search(r"\b(history|medieval|ancient|century|dynasty|empire|origin)\b", full_l):
                    recency_scores[i] = -0.60

        # Base composite formula
        if analysis.target_date:
            # When an explicit date is requested: Date relevance is paramount!
            final_scores = (
                0.32 * date_scores
                + 0.20 * tfidf_norm
                + 0.18 * bm25_norm
                + 0.08 * char_norm
                + 0.08 * coverage
                + 0.06 * title_scores
                + 0.04 * entity_scores
                + 0.04 * source_scores
            )
        elif analysis.is_latest or analysis.intent == "latest_news":
            final_scores = (
                0.32 * recency_scores
                + 0.22 * tfidf_norm
                + 0.18 * bm25_norm
                + 0.10 * coverage
                + 0.08 * char_norm
                + 0.04 * title_scores
                + 0.04 * entity_scores
                + 0.02 * source_scores
            )
        else:
            final_scores = (
                0.28 * tfidf_norm
                + 0.24 * bm25_norm
                + 0.12 * char_norm
                + 0.12 * coverage
                + 0.08 * title_scores
                + 0.06 * entity_scores
                + 0.05 * phrase_scores
                + 0.03 * intent_scores
                + 0.02 * source_scores
            )

    # Normalize final scores to [0, 1]
    final_scores = normalize_scores(final_scores)

    ranked_indices = np.argsort(final_scores)[::-1]
    top_indices = ranked_indices[:max(1, top_k)]

    candidates = [(records[idx], float(final_scores[idx])) for idx in top_indices]
    return candidates, final_scores
