"""
core/answer.py
Grounded answer generation, category-based news summarization,
calibrated confidence calculation, and source formatting.
Strictly zero hallucination and zero contamination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.query_understanding import QueryAnalysis, TargetDate, normalize_text, tokenize
from core.reranking import RerankedPassage

CATEGORY_KEYWORDS = {
    "Conflict": ["war", "conflict", "drone", "missile", "strike", "attack", "military", "troops", "ukraine", "russia", "gaza", "israel", "clashes", "arrested", "protests"],
    "Politics": ["election", "parliament", "congress", "president", "prime minister", "government", "policy", "minister", "senate", "midterm", "budget", "chancellor"],
    "Sports": ["cup", "cricket", "football", "soccer", "tennis", "basketball", "match", "tournament", "score", "won", "winner", "champion", "lafc", "galaxy", "premier league"],
    "India": ["india", "delhi", "mumbai", "indian", "modi", "bengaluru", "the hindu", "ndtv", "times of india"],
    "Technology": ["ai", "lab", "software", "starlink", "spacex", "musk", "biosecurity", "tech", "computing", "algorithm", "data"],
    "World": ["world", "international", "un", "global", "europe", "us", "uk", "france", "asia"],
}


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    selected_passages: List[Dict]
    sources_formatted: List[Dict]
    is_confident: bool


def categorize_sentence(sentence: str, default_category: str = "General") -> str:
    """Classify a news sentence into high-level news categories."""
    s_low = sentence.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in s_low for kw in kws):
            return cat
    return default_category


def calculate_calibrated_confidence(
    top_score: float,
    analysis: QueryAnalysis,
    top_passage: Dict,
    supporting_passages: List[Dict]
) -> float:
    """
    Calculate a grounded, multi-factor confidence score based on:
      - Top retrieval similarity score (0.0 to 1.0)
      - Query keyword coverage in winning passage
      - Source quality tier of winning passage
      - Exact date alignment (if applicable)
      - Corroboration across multiple sources
    """
    if top_score <= 0 or not top_passage:
        return 0.0

    content = top_passage.get("content", "") + " " + top_passage.get("title", "")
    content_toks = set(tokenize(content))
    q_toks = set(analysis.keywords)
    coverage = len(q_toks & content_toks) / max(len(q_toks), 1)

    # Source quality
    quality = float(top_passage.get("quality_score", 0.70))

    # Date alignment
    date_match = 0.5
    if analysis.target_date:
        t_iso = analysis.target_date.iso_date
        t_str = analysis.target_date.display_str.lower()
        rec_date = top_passage.get("pub_date", "") or top_passage.get("detected_date", "")
        if t_iso in rec_date or t_str in content.lower():
            date_match = 1.0
        else:
            date_match = 0.2

    # Agreement / supporting count
    support_bonus = min(0.20, len(supporting_passages) * 0.05)

    raw_conf = (
        0.35 * top_score
        + 0.25 * coverage
        + 0.15 * quality
        + 0.15 * date_match
        + 0.10 * (1.0 if support_bonus > 0 else 0.5)
    )
    return round(float(np.clip(raw_conf, 0.05, 0.98)), 2)


def format_news_summary_answer(
    target_date: TargetDate,
    reranked: List[RerankedPassage]
) -> str:
    """
    Format a multi-category structured news summary for daily news queries:
      On 25 October 2026, major developments included:
      1. **Conflict:** ...
      2. **Politics:** ...
    """
    date_str = target_date.display_str
    lines = [f"On {date_str}, major developments included:\n"]

    used_categories = set()
    category_items = []

    # Map passages into distinct categories
    for rp in reranked:
        if not rp.best_sentences:
            continue
        primary_sent = rp.best_sentences[0]
        cat = categorize_sentence(primary_sent, rp.record.get("category", "World"))
        if cat in used_categories:
            continue

        used_categories.add(cat)
        category_items.append((cat, primary_sent))
        if len(category_items) >= 5:
            break

    # If only 1 category found or general, format nicely
    if not category_items:
        # Fallback to top sentences
        for idx, rp in enumerate(reranked[:3], start=1):
            if rp.best_sentences:
                lines.append(f"{idx}. {rp.best_sentences[0]}")
        return "\n".join(lines)

    for idx, (cat, sent) in enumerate(category_items, start=1):
        lines.append(f"{idx}. **{cat}:** {sent}")

    return "\n".join(lines)


def generate_grounded_answer(
    reranked: List[RerankedPassage],
    analysis: QueryAnalysis,
    confidence_threshold: float = 0.12
) -> AnswerResult:
    """
    Generate grounded extractive answer strictly using selected relevant evidence.
    Zero hallucination.
    """
    if not reranked:
        return AnswerResult(
            answer="I couldn't find enough reliable evidence to answer this confidently.",
            confidence=0.0,
            selected_passages=[],
            sources_formatted=[],
            is_confident=False
        )

    # Filter to genuinely relevant passages
    relevant_passages = [rp for rp in reranked if rp.is_relevant]
    if not relevant_passages:
        # Check if top passage is at least moderately relevant
        if reranked[0].rerank_score >= confidence_threshold:
            relevant_passages = [reranked[0]]
        else:
            return AnswerResult(
                answer="I couldn't find enough reliable evidence to answer this confidently.",
                confidence=float(reranked[0].rerank_score),
                selected_passages=[],
                sources_formatted=[],
                is_confident=False
            )

    top_rp = relevant_passages[0]
    top_record = top_rp.record
    confidence = calculate_calibrated_confidence(
        top_rp.rerank_score,
        analysis,
        top_record,
        [rp.record for rp in relevant_passages[1:]]
    )

    if confidence < confidence_threshold:
        return AnswerResult(
            answer="I couldn't find enough reliable evidence to answer this confidently.",
            confidence=confidence,
            selected_passages=[top_record],
            sources_formatted=[],
            is_confident=False
        )

    # 1. News summary format
    if analysis.intent == "daily_news_summary" and analysis.target_date:
        answer_text = format_news_summary_answer(analysis.target_date, relevant_passages)

    # 2. General / Factual / Definition format
    else:
        # Select best sentences strictly from winning passage
        chosen_sentences = list(top_rp.best_sentences)

        # In rare cases where a second passage adds a directly related verified fact on the SAME topic
        if len(chosen_sentences) < 2 and len(relevant_passages) > 1:
            second_rp = relevant_passages[1]
            if second_rp.best_sentences:
                candidate_sent = second_rp.best_sentences[0]
                # Verify topic purity: candidate must share major keywords
                s_toks = set(tokenize(candidate_sent))
                q_toks = set(analysis.keywords)
                if len(s_toks & q_toks) >= max(1, len(q_toks) // 2):
                    if candidate_sent.lower() not in " ".join(chosen_sentences).lower():
                        chosen_sentences.append(candidate_sent)

        answer_text = " ".join(chosen_sentences).strip()
        if not answer_text:
            answer_text = top_record.get("content", "")

    # Format sources
    sources_formatted = []
    seen_sources = set()
    for rp in relevant_passages:
        rec = rp.record
        title = rec.get("title", "Untitled Source")
        pub = rec.get("publisher", rec.get("category", "Knowledge Base"))
        date = rec.get("pub_date", "") or rec.get("detected_date", "") or rec.get("detected_date_str", "")
        url = rec.get("url", "")
        key = (title.lower(), pub.lower())
        if key in seen_sources:
            continue
        seen_sources.add(key)

        sources_formatted.append({
            "title": title,
            "publisher": pub,
            "pub_date": date,
            "url": url,
            "score": rp.rerank_score,
            "category": rec.get("category", "")
        })

    return AnswerResult(
        answer=answer_text,
        confidence=confidence,
        selected_passages=[rp.record for rp in relevant_passages],
        sources_formatted=sources_formatted,
        is_confident=True
    )
