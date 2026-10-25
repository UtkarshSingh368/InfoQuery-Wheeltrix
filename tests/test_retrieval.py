"""
tests/test_retrieval.py
Unit tests for Good (TF-IDF), Better (BM25), and Best (Hybrid Multi-Signal Ensemble)
retrieval algorithms.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from core.query_understanding import analyze_query
from core.retrieval import (
    calculate_bm25_scores,
    calculate_date_scores,
    normalize_scores,
    retrieve_candidates,
)


class TestRetrievalAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kb_path = Path(__file__).parents[1] / "data" / "knowledge_base.json"
        with open(kb_path, "r", encoding="utf-8") as f:
            cls.records = json.load(f)

    def test_tfidf_good_mode(self):
        analysis = analyze_query("What is TF-IDF?")
        candidates, scores = retrieve_candidates(
            question="What is TF-IDF?",
            records=self.records,
            analysis=analysis,
            method="TF-IDF + Cosine Similarity",
            top_k=3
        )
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0][0]["title"], "TF-IDF")
        self.assertGreater(candidates[0][1], 0.3)

    def test_bm25_better_mode(self):
        analysis = analyze_query("What is BM25?")
        candidates, scores = retrieve_candidates(
            question="What is BM25?",
            records=self.records,
            analysis=analysis,
            method="BM25",
            top_k=3
        )
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0][0]["title"], "BM25")
        self.assertGreater(candidates[0][1], 0.3)

    def test_hybrid_mode(self):
        analysis = analyze_query("What is machine learning?")
        candidates, scores = retrieve_candidates(
            question="What is machine learning?",
            records=self.records,
            analysis=analysis,
            method="Hybrid (TF-IDF + BM25)",
            top_k=3
        )
        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0][0]["title"], "Machine Learning")

    def test_best_ensemble_mode(self):
        analysis = analyze_query("Who won the 2026 FIFA World Cup?")
        candidates, scores = retrieve_candidates(
            question="Who won the 2026 FIFA World Cup?",
            records=self.records,
            analysis=analysis,
            method="Best — Hybrid + Reranking",
            top_k=3
        )
        self.assertGreater(len(candidates), 0)
        self.assertIn("2026 FIFA World Cup", candidates[0][0]["title"])
        self.assertIn("Spain", candidates[0][0]["content"])

    def test_normalize_scores_edge_cases(self):
        # Empty array
        empty = np.array([])
        self.assertEqual(len(normalize_scores(empty)), 0)

        # All zeros
        zeros = np.array([0.0, 0.0, 0.0])
        self.assertTrue(np.all(normalize_scores(zeros) == 0.0))

        # Normal positive values
        vals = np.array([10.0, 20.0, 30.0])
        normed = normalize_scores(vals)
        self.assertAlmostEqual(normed[0], 0.0)
        self.assertAlmostEqual(normed[2], 1.0)


if __name__ == "__main__":
    unittest.main()
