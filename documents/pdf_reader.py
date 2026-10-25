"""
documents/pdf_reader.py
Extract text from PDF documents using pypdf.
"""

from __future__ import annotations

import io
from typing import BinaryIO, Union

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def read_pdf(file_input: Union[BinaryIO, bytes]) -> str:
    """Extract all text pages from a PDF file."""
    if PdfReader is None:
        raise RuntimeError("PDF support is unavailable. Please install pypdf.")

    if isinstance(file_input, bytes):
        file_input = io.BytesIO(file_input)

    reader = PdfReader(file_input)
    pages_text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text.strip())

    if not pages_text:
        return ""
    return "\n\n".join(pages_text)
