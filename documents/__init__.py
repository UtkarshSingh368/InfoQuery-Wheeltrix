"""
Documents package for reading PDF, DOCX, TXT files and date-aware passage chunking.
"""

from documents.pdf_reader import read_pdf
from documents.docx_reader import read_docx
from documents.text_reader import read_text
from documents.chunker import chunk_document, clean_document_text

__all__ = [
    "read_pdf",
    "read_docx",
    "read_text",
    "chunk_document",
    "clean_document_text",
]
