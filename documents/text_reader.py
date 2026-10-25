"""
documents/text_reader.py
Extract and clean text from plain text files.
"""

from __future__ import annotations

from typing import Union


def read_text(file_input: Union[bytes, str]) -> str:
    """Decode raw bytes into text with multi-encoding fallback."""
    if isinstance(file_input, str):
        return file_input

    for encoding in ["utf-8", "latin-1", "windows-1252", "ascii"]:
        try:
            return file_input.decode(encoding)
        except UnicodeDecodeError:
            continue

    return file_input.decode("utf-8", errors="ignore")
