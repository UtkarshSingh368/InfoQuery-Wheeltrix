"""
tests/test_documents.py
Unit tests for PDF, DOCX, and TXT document readers and date-aware chunking.
"""

import io
import unittest

from documents.chunker import chunk_document, clean_document_text
from documents.docx_reader import read_docx
from documents.pdf_reader import read_pdf
from documents.text_reader import read_text


class TestDocumentProcessing(unittest.TestCase):
    def test_text_reader_encodings(self):
        utf8_bytes = "Machine learning is a subset of AI.".encode("utf-8")
        self.assertEqual(read_text(utf8_bytes), "Machine learning is a subset of AI.")

        latin1_bytes = "Crème brûlée and natural language processing.".encode("latin-1")
        self.assertIn("language processing", read_text(latin1_bytes))

    def test_chunker_removes_instructions_and_boilerplate(self):
        raw_doc = """
        Confidential
        Page 1 of 5
        Instructions: Please review the following report before submission.

        On 19 October 2026, the global summit on artificial intelligence concluded in Geneva.
        Representatives from 40 nations signed the international safety accord.

        Demo Document
        Copyright 2026
        """
        cleaned = clean_document_text(raw_doc)
        self.assertNotIn("Page 1 of 5", cleaned)
        self.assertNotIn("Confidential", cleaned)
        self.assertNotIn("Instructions:", cleaned)
        self.assertIn("19 October 2026", cleaned)

        chunks = chunk_document(raw_doc, "AI_Summit_Report.txt")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["detected_date"], "2026-10-19")
        self.assertIn("Geneva", chunks[0]["content"])

    def test_docx_reader_mock(self):
        # Create a real in-memory docx if python-docx is available
        try:
            import docx
            doc = docx.Document()
            doc.add_heading("Quantum Computing 2026", 0)
            doc.add_paragraph("Quantum computers achieved a 10,000 qubit milestone on 15 October 2026.")
            buf = io.BytesIO()
            doc.save(buf)
            buf.seek(0)

            extracted = read_docx(buf.getvalue())
            self.assertIn("Quantum Computing 2026", extracted)
            self.assertIn("10,000 qubit milestone", extracted)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
