"""
qa_engine.py
Unified Q&A Engine for InfoQuery.
Provides backward-compatible interface and connects Query Understanding,
Retrieval (TF-IDF, BM25, Best Ensemble), Passage Reranking, and Grounded Answers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.answer import AnswerResult, generate_grounded_answer
from core.query_understanding import (
    QueryAnalysis,
    analyze_query,
    extract_date,
    normalize_text,
    tokenize,
)
from core.reranking import rerank_candidate_passages
from core.retrieval import (
    build_bm25_index,
    build_tfidf_index,
    calculate_bm25_scores,
    normalize_scores,
    retrieve_candidates,
)


# Legacy backward-compatibility aliases
def normalize(text: str) -> str:
    return normalize_text(text)


def build_index(records: List[Dict]):
    return build_tfidf_index(records)


def bm25_scores(query: str, records: List[Dict], k1: float = 1.5, b: float = 0.75):
    return calculate_bm25_scores(query, records, k1=k1, b=b)


def query_terms(question: str):
    from core.query_understanding import STOP_WORDS
    return {t for t in tokenize(question) if len(t) > 2 and t not in STOP_WORDS}


def question_intent(question: str) -> str:
    analysis = analyze_query(question)
    return analysis.intent


def best_answer_sentences(question: str, record: Dict, vectorizer=None, max_sentences: int = 2) -> str:
    """Extract winning sentences from a single passage."""
    analysis = analyze_query(question)
    candidates = [(record, 1.0)]
    reranked = rerank_candidate_passages(candidates, analysis, top_k=1)
    if reranked and reranked[0].best_sentences:
        return " ".join(reranked[0].best_sentences[:max_sentences])
    return record.get("content", "")


def answer_question(
    question: str,
    records: List[Dict],
    method: str = "Best — Hybrid + Reranking",
    top_k: int = 3,
    threshold: float = 0.12,
    return_details: bool = False
) -> Union[Tuple[str, float, List[Tuple[Dict, float]]], Tuple[AnswerResult, List[Tuple[Dict, float]]]]:
    """
    Core Q&A answering pipeline:
      1. Query Understanding (intent, dates, entities, keywords)
      2. Candidate Retrieval (TF-IDF / BM25 / Best Hybrid Ensemble)
      3. Passage & Sentence Reranking (cross-signal, date-aware, purity checks)
      4. Grounded Answer Generation (zero hallucination, structured formats)

    Returns:
      (answer_text, confidence_score, ranked_results_list)
      or (AnswerResult, ranked_results_list) if return_details=True
    """
    if not records:
        empty_res = AnswerResult(
            answer="I couldn't find enough reliable evidence to answer this confidently.",
            confidence=0.0,
            selected_passages=[],
            sources_formatted=[],
            is_confident=False
        )
        return (empty_res, []) if return_details else (empty_res.answer, 0.0, [])

    # Step 1: Query Understanding
    analysis = analyze_query(question)

    # Step 2: Retrieval (TF-IDF, BM25, or Best Hybrid)
    candidates, all_scores = retrieve_candidates(
        question=question,
        records=records,
        analysis=analysis,
        method=method,
        top_k=max(top_k * 2, 6)
    )

    if not candidates:
        empty_res = AnswerResult(
            answer="I couldn't find enough reliable evidence to answer this confidently.",
            confidence=0.0,
            selected_passages=[],
            sources_formatted=[],
            is_confident=False
        )
        return (empty_res, []) if return_details else (empty_res.answer, 0.0, [])

    # Step 3: Passage & Sentence Reranking
    reranked = rerank_candidate_passages(candidates, analysis, top_k=top_k)

    # Step 4: Grounded Answer Generation
    answer_res = generate_grounded_answer(reranked, analysis, confidence_threshold=threshold)

    # Format results list for compatibility: [(record, score), ...]
    results_list = [(rp.record, float(rp.rerank_score)) for rp in reranked]

    if return_details:
        return answer_res, results_list

    return answer_res.answer, answer_res.confidence, results_list
