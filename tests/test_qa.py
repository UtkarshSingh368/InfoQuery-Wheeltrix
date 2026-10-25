"""
tests/test_qa.py
Core end-to-end acceptance tests matching Section 21 of requirements:
  - Test 1: Machine Learning (topic pure, no football/basketball contamination)
  - Test 2: BM25 explanation
  - Test 3: 25 October 2026 date-specific query
  - Test 4: PDF upload with multiple dates; 19 October 2026 query ranks 19 October above other dates
  - Test 5: Latest news about Ukraine (recent events over generic background)
  - Test 6: 2024 ICC Men's T20 World Cup winner (India)
  - Test 7: 2026 FIFA World Cup winner (Spain)
  - Test 8: Network failure / invalid web page (graceful fallback, zero crash)
"""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from documents.chunker import chunk_document
from documents.pdf_reader import read_pdf
from qa_engine import answer_question
from web.extraction import extract_article_content
from web.search import perform_live_web_search


def create_minimal_pdf_bytes(pages_text: list) -> bytes:
    """Generate a valid multi-entry PDF containing dated news items."""
    stream_parts = []
    for line in pages_text:
        clean_line = line.replace("(", "[").replace(")", "]")
        stream_parts.append(f"({clean_line}) Tj T*")
    stream_text = "BT /F1 11 Tf 40 750 Td " + " ".join(stream_parts) + " ET"
    stream_len = len(stream_text.encode("latin-1"))

    pdf_str = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length {stream_len} >>
stream
{stream_text}
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000300 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
450
%%EOF"""
    return pdf_str.encode("latin-1")


class TestInfoQueryAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kb_path = Path(__file__).parents[1] / "data" / "knowledge_base.json"
        with open(kb_path, "r", encoding="utf-8") as f:
            cls.kb_records = json.load(f)

    # Test 1: Machine Learning must be topic pure (no football/basketball contamination)
    def test_test1_machine_learning_topic_pure(self):
        answer, score, results = answer_question(
            "What is machine learning?",
            self.kb_records,
            method="Best — Hybrid + Reranking"
        )
        self.assertIn("artificial intelligence", answer.lower())
        self.assertNotIn("basketball", answer.lower())
        self.assertNotIn("football", answer.lower())
        self.assertNotIn("offside", answer.lower())
        self.assertEqual(results[0][0]["title"], "Machine Learning")
        self.assertGreater(score, 0.40)

    # Test 2: BM25 explanation
    def test_test2_bm25_explanation(self):
        answer, score, results = answer_question(
            "What is BM25?",
            self.kb_records,
            method="Best — Hybrid + Reranking"
        )
        self.assertTrue(
            "ranking algorithm" in answer.lower() or "information retrieval" in answer.lower()
        )
        self.assertEqual(results[0][0]["title"], "BM25")
        self.assertGreater(score, 0.40)

    # Test 3: What happened on 25 October 2026?
    def test_test3_date_specific_25_october_2026(self):
        test_records = [
            {
                "title": "LA Galaxy vs LAFC Match Summary",
                "publisher": "Sports Daily",
                "pub_date": "2026-10-25",
                "content": "On 25 October 2026, LA Galaxy and LAFC played an intense playoff derby in front of 22,000 fans.",
                "category": "Sports"
            },
            {
                "title": "Diplomatic Summit Opening",
                "publisher": "World Wire",
                "pub_date": "2026-10-25",
                "content": "On 25 October 2026, world leaders convened in Geneva to ratify the new international AI safety protocol.",
                "category": "World"
            },
            {
                "title": "Previous Week Space Launch",
                "publisher": "Aero News",
                "pub_date": "2026-10-18",
                "content": "On 18 October 2026, a commercial communications satellite launched from Cape Canaveral.",
                "category": "Tech"
            }
        ]
        answer, score, results = answer_question(
            "What happened on 25 October 2026?",
            test_records,
            method="Best — Hybrid + Reranking"
        )
        # Winning evidence must be specifically associated with 25 October 2026
        top_record = results[0][0]
        self.assertEqual(top_record.get("pub_date"), "2026-10-25")
        self.assertIn("25 October 2026", answer)
        # 18 October news should not outrank 25 October news
        self.assertNotEqual(results[0][0]["title"], "Previous Week Space Launch")

    # Test 4: Upload a PDF containing several dated news items; 19 October ranks above other dates
    def test_test4_pdf_date_aware_ranking(self):
        pdf_lines = [
            "News Bulletin 15 October 2026: Global markets rallied as inflation data cooled.",
            "News Bulletin 19 October 2026: European Space Agency successfully launched the Hera asteroid probe.",
            "News Bulletin 22 October 2026: Tech summit opened in Tokyo with focus on quantum encryption."
        ]
        pdf_bytes = create_minimal_pdf_bytes(pdf_lines)
        extracted_text = read_pdf(pdf_bytes)
        chunks = chunk_document(extracted_text, "Daily_Bulletin.pdf")

        self.assertGreaterEqual(len(chunks), 1)

        answer, score, results = answer_question(
            "What happened on 19 October 2026?",
            chunks,
            method="Best — Hybrid + Reranking"
        )
        self.assertIn("19 October 2026", results[0][0]["content"])
        self.assertIn("Hera", results[0][0]["content"])

    # Test 5: Latest news about Ukraine
    def test_test5_latest_news_ukraine(self):
        ukraine_records = [
            {
                "title": "Starlink Speeds and Drone Defense Updates",
                "publisher": "CBS News",
                "pub_date": "2026-10-25",
                "content": "Ukraine military officials announced restored communication links and reported drone defense measures on 25 October 2026.",
                "category": "Live Web • CBS News"
            },
            {
                "title": "History of Ukraine Early Medieval Period",
                "publisher": "Wikipedia",
                "pub_date": "",
                "content": "Kievan Rus was a medieval political federation in Belarus, Russia, and Ukraine from the late 9th to mid-13th century.",
                "category": "Live Web • Wikipedia"
            }
        ]
        answer, score, results = answer_question(
            "latest news about Ukraine",
            ukraine_records,
            method="Best — Hybrid + Reranking"
        )
        # Must select the current news record rather than generic medieval history
        self.assertEqual(results[0][0]["title"], "Starlink Speeds and Drone Defense Updates")
        self.assertIn("drone", answer.lower())

    # Test 6: Who won the 2024 ICC Men's T20 World Cup?
    def test_test6_t20_world_cup_winner(self):
        answer, score, results = answer_question(
            "Who won the 2024 ICC Men's T20 World Cup?",
            self.kb_records,
            method="Best — Hybrid + Reranking"
        )
        self.assertIn("India won the 2024 ICC Men's T20 World Cup", answer)
        self.assertIn("India", answer)
        top_title = results[0][0]["title"]
        self.assertTrue(
            top_title in ["2024 ICC Men's T20 World Cup", "Cricket — T20 World Cup 2024"] or "T20 World Cup" in top_title
        )

    # Test 7: Who won the 2026 FIFA World Cup?
    def test_test7_fifa_2026_winner(self):
        answer, score, results = answer_question(
            "Who won the 2026 FIFA World Cup?",
            self.kb_records,
            method="Best — Hybrid + Reranking"
        )
        self.assertIn("Spain won the 2026 FIFA World Cup", answer)
        self.assertIn("Spain", answer)
        self.assertEqual(results[0][0]["title"], "2026 FIFA World Cup")

    # Test 8: No internet / failed web page (graceful fallback, zero crash)
    def test_test8_network_failure_fallback(self):
        # 1. Test extraction with unreachable URL
        article = extract_article_content(
            url="http://invalid-non-existent-domain-12345.org/article",
            default_title="Fallback Title",
            default_publisher="Fallback Pub",
            fallback_snippet="Fallback text snippet content.",
            timeout=1
        )
        self.assertTrue(article.fallback_used)
        self.assertEqual(article.title, "Fallback Title")
        self.assertIn("Fallback text snippet", article.body)

        # 2. Test search execution when requests raise network error
        with patch("requests.get", side_effect=Exception("ConnectionRefused: No internet")):
            from core.query_understanding import analyze_query
            analysis = analyze_query("What happened on 25 October 2026?")
            # Must return empty list or fallback, not raise an unhandled exception
            res = perform_live_web_search("What happened on 25 October 2026?", analysis)
            self.assertIsInstance(res, list)


if __name__ == "__main__":
    unittest.main()
