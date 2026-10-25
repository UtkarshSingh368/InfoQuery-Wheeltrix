"""
tests/test_query_understanding.py
Unit tests for date extraction, question intent detection, entity parsing,
and multi-query search generation.
"""

import unittest

from core.query_understanding import (
    analyze_query,
    detect_intent,
    extract_date,
    extract_entities,
    generate_search_queries,
)


class TestQueryUnderstanding(unittest.TestCase):
    def test_date_extraction_formats(self):
        # Format 1: 25 October 2026
        d1 = extract_date("What happened on 25 October 2026?")
        self.assertIsNotNone(d1)
        self.assertEqual(d1.iso_date, "2026-10-25")
        self.assertEqual(d1.day, 25)
        self.assertEqual(d1.month, 10)
        self.assertEqual(d1.year, 2026)

        # Format 2: 19th July 2026
        d2 = extract_date("The final took place on 19th July 2026 in New Jersey.")
        self.assertIsNotNone(d2)
        self.assertEqual(d2.iso_date, "2026-07-19")

        # Format 3: October 25, 2026
        d3 = extract_date("October 25, 2026 major events")
        self.assertIsNotNone(d3)
        self.assertEqual(d3.iso_date, "2026-10-25")

        # Format 4: ISO 2026-10-25
        d4 = extract_date("Events of 2026-10-25")
        self.assertIsNotNone(d4)
        self.assertEqual(d4.iso_date, "2026-10-25")

    def test_intent_detection(self):
        # Daily news summary
        a1 = analyze_query("What happened on 25 October 2026?")
        self.assertEqual(a1.intent, "daily_news_summary")
        self.assertIsNotNone(a1.target_date)

        # Definition
        a2 = analyze_query("What is machine learning?")
        self.assertEqual(a2.intent, "definition")

        # Sports
        a3 = analyze_query("Who won the 2024 ICC Men's T20 World Cup?")
        self.assertEqual(a3.intent, "sports")

        # Conflict
        a4 = analyze_query("latest news about Ukraine war conflict")
        self.assertTrue(a4.is_latest)

    def test_search_query_generation(self):
        a = analyze_query("What happened on 25 October 2026?")
        self.assertGreaterEqual(len(a.search_queries), 3)
        # Should generate diverse queries covering India, World, Sports, Politics
        combined = " ".join(a.search_queries).lower()
        self.assertIn("25 october 2026", combined)
        self.assertIn("news", combined)

    def test_entity_extraction(self):
        entities = extract_entities("Who won the FIFA World Cup 2026?")
        self.assertTrue(any("FIFA" in e for e in entities))


if __name__ == "__main__":
    unittest.main()
