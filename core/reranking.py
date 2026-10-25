"""
core/reranking.py
Passage and sentence-level reranker.
Applies cross-signal scoring, topic purity constraints, date alignment,
and prevents contamination across unrelated knowledge categories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from core.query_understanding import QueryAnalysis, TargetDate, extract_date, normalize_text, tokenize

# Known topic domain clusters to prevent cross-domain contamination
TOPIC_POLLUTANTS = {
    "tech": {"football", "basketball", "soccer", "cricket", "tennis", "olympic", "offside", "touchdown", "free throw"},
    "sports": {"algorithm", "neural network", "database", "photosynthesis", "cloud computing"},
    "science": {"nfl", "nba", "ipl", "fifa", "premier league"},
}


@dataclass
class ScoredSentence:
    text: str
    score: float
    sentence_idx: int
    is_pure: bool
    matched_terms: Set[str]


@dataclass
class RerankedPassage:
    record: Dict
    initial_score: float
    rerank_score: float
    best_sentences: List[str]
    is_relevant: bool


def split_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly."""
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = []
    for s in raw:
        cleaned = normalize_text(s)
        if len(cleaned) >= 15:
            sentences.append(cleaned)
    return sentences or ([normalize_text(text)] if text else [])


def score_sentence(
    sentence: str,
    analysis: QueryAnalysis,
    record: Dict,
    sentence_idx: int,
    total_sentences: int
) -> ScoredSentence:
    """
    Score an individual sentence for topical relevance, intent match,
    date alignment, and topic purity.
    """
    s_lower = sentence.lower()
    s_toks = set(tokenize(sentence))
    q_toks = set(analysis.keywords)

    overlap = len(q_toks & s_toks) / max(len(q_toks), 1)

    # 1. Topic Purity / Contamination check
    is_pure = True
    if analysis.intent in ["definition", "technology", "science"]:
        pollutants = TOPIC_POLLUTANTS.get("tech", set())
        # If query doesn't ask about sports but sentence mentions sports terms
        if not (set(analysis.keywords) & pollutants):
            found_pollutants = s_toks & pollutants
            if found_pollutants:
                is_pure = False
                overlap *= 0.10  # severe downweight

    # 2. Date alignment
    date_score = 0.0
    if analysis.target_date:
        t_iso = analysis.target_date.iso_date
        t_str = analysis.target_date.display_str.lower()
        t_day = str(analysis.target_date.day)
        t_month = analysis.target_date.month_name.lower()
        t_year = str(analysis.target_date.year)

        # Check explicit sentence date
        if t_str in s_lower or t_iso in s_lower:
            date_score = 0.40
        elif t_month in s_lower and t_year in s_lower:
            if re.search(rf"\b{t_day}(?:st|nd|rd|th)?\s+{t_month}\b|\b{t_month}\s+{t_day}\b", s_lower):
                date_score = 0.35
            else:
                date_score = 0.10
        elif record.get("pub_date") == t_iso or record.get("detected_date") == t_iso:
            date_score = 0.25

        # Conflicting date penalty
        sent_date = extract_date(sentence)
        if sent_date and sent_date.iso_date != t_iso:
            date_score -= 0.30

    elif analysis.is_latest or analysis.intent == "latest_news":
        pub = record.get("pub_date", "") or record.get("detected_date", "")
        if pub:
            date_score = 0.35
        if re.search(r"\b(medieval|ancient|century|dynasty|empire)\b", s_lower):
            date_score -= 0.50

    # 3. Exact phrase match
    phrase_score = 0.0
    clean_q = analysis.clean_question.lower()
    if clean_q and clean_q in s_lower:
        phrase_score = 0.25

    # 4. Intent specific bonus
    intent_bonus = 0.0
    if analysis.intent == "definition":
        if re.search(r"\b(is|are|refers to|means|stands for|defined as)\b", s_lower):
            intent_bonus = 0.20
    elif analysis.intent == "sports":
        if re.search(r"\b(won|winner|champion|defeated|final|score|player of the)\b", s_lower):
            intent_bonus = 0.18
    elif analysis.intent in ["daily_news_summary", "latest_news"]:
        if re.search(r"\b(reported|said|announced|launched|signed|held|met|killed|won|struck)\b", s_lower):
            intent_bonus = 0.12

    # 5. Position bias: lead sentences in news/articles often contain the core fact
    position_bonus = 0.05 if sentence_idx == 0 else 0.0

    final_score = (
        0.45 * overlap
        + 0.25 * date_score
        + 0.15 * phrase_score
        + 0.10 * intent_bonus
        + position_bonus
    )
    if not is_pure:
        final_score -= 0.40

    return ScoredSentence(
        text=sentence,
        score=max(0.0, final_score),
        sentence_idx=sentence_idx,
        is_pure=is_pure,
        matched_terms=q_toks & s_toks
    )


def rerank_candidate_passages(
    candidates: List[Tuple[Dict, float]],
    analysis: QueryAnalysis,
    top_k: int = 5
) -> List[RerankedPassage]:
    """
    Performs sentence-level inspection on candidate passages to:
      1. Prevent cross-topic contamination
      2. Enforce date alignment
      3. Extract winning coherent sentences
      4. Adjust passage score based on sentence-level evidence
    """
    reranked: List[RerankedPassage] = []

    for record, init_score in candidates:
        content = record.get("content", "")
        sentences = split_sentences(content)
        if not sentences:
            continue

        scored_sentences = [
            score_sentence(s, analysis, record, idx, len(sentences))
            for idx, s in enumerate(sentences)
        ]

        # Filter out impure or heavily penalized sentences
        pure_sentences = [ss for ss in scored_sentences if ss.is_pure]
        pool = pure_sentences if pure_sentences else scored_sentences

        pool.sort(key=lambda x: x.score, reverse=True)
        best_sentence = pool[0]

        chosen = [best_sentence.text]
        # Allow one closely related adjacent sentence if it provides critical context
        if len(pool) > 1 and len(sentences) > 1:
            for cand in pool[1:]:
                if abs(cand.sentence_idx - best_sentence.sentence_idx) == 1:
                    if cand.score >= 0.18 and cand.is_pure and cand.text != best_sentence.text:
                        # Append in natural order
                        if cand.sentence_idx < best_sentence.sentence_idx:
                            chosen.insert(0, cand.text)
                        else:
                            chosen.append(cand.text)
                        break

        # Passage rerank score combines retrieval score and top sentence score
        top_sent_score = best_sentence.score
        # Check date / recency alignment for passage
        date_boost = 0.0
        if analysis.target_date:
            rec_date = record.get("pub_date", "") or record.get("detected_date", "")
            if analysis.target_date.iso_date in rec_date:
                date_boost = 0.20
            elif rec_date and rec_date != analysis.target_date.iso_date:
                date_boost = -0.20
        elif analysis.is_latest or analysis.intent == "latest_news":
            rec_date = record.get("pub_date", "") or record.get("detected_date", "")
            if rec_date:
                date_boost = 0.25
            full_text = f"{record.get('title', '')} {record.get('content', '')}".lower()
            if re.search(r"\b(history|medieval|ancient|century|dynasty)\b", full_text):
                date_boost = -0.40

        rerank_score = (
            0.50 * init_score
            + 0.40 * top_sent_score
            + date_boost
        )

        is_relevant = (rerank_score >= 0.15) and best_sentence.is_pure and (len(best_sentence.matched_terms) > 0 or init_score > 0.40)

        reranked.append(RerankedPassage(
            record=record,
            initial_score=init_score,
            rerank_score=float(rerank_score),
            best_sentences=chosen,
            is_relevant=is_relevant
        ))

    # Sort descending by rerank_score
    reranked.sort(key=lambda x: x.rerank_score, reverse=True)
    return reranked[:top_k]
