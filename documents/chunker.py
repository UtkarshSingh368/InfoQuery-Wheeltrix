"""
documents/chunker.py
Cleans uploaded document text, filters out document metadata/instructions boilerplate,
and chunks text into cohesive, date-aware passages.
"""

from __future__ import annotations

import re
from typing import Dict, List

from core.query_understanding import extract_date, normalize_text

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*page \d+\s*(of \d+)?\s*$", re.I),
    re.compile(r"^\s*confidential\s*$", re.I),
    re.compile(r"^\s*(all rights reserved|copyright \d{4})\s*$", re.I),
    re.compile(r"^\s*instructions\s*:\s*.*$", re.I),
    re.compile(r"^\s*demo document\s*$", re.I),
    re.compile(r"^\s*table of contents\s*$", re.I),
]


def clean_document_text(text: str) -> str:
    """Strip out document header/footer artifacts and normalize whitespace."""
    lines = text.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue
        # Check against boilerplate lines
        if any(p.match(stripped) for p in BOILERPLATE_PATTERNS):
            continue
        filtered_lines.append(stripped)

    return "\n".join(filtered_lines)


def chunk_document(text: str, source_name: str) -> List[Dict]:
    """
    Split document text into clean, searchable passages.
    Extracts dates associated with each chunk for date-aware retrieval.
    """
    cleaned = clean_document_text(text)
    # Split on multiple newlines or logical paragraph separators
    raw_blocks = re.split(r"\n\s*\n", cleaned)

    chunks = []
    chunk_num = 1

    for block in raw_blocks:
        norm_block = normalize_text(block)
        if len(norm_block) < 30:
            continue

        # If block is exceptionally long (>900 chars), subdivide into sentences
        if len(norm_block) > 900:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", norm_block)
            current = []
            current_len = 0
            for sent in sentences:
                current.append(sent)
                current_len += len(sent)
                if current_len >= 500:
                    sub_text = " ".join(current).strip()
                    d = extract_date(sub_text)
                    chunks.append({
                        "title": source_name,
                        "category": f"Uploaded • {source_name}",
                        "content": sub_text,
                        "id": f"{source_name}-{chunk_num}",
                        "detected_date": d.iso_date if d else "",
                        "detected_date_str": d.display_str if d else "",
                        "publisher": source_name,
                        "url": ""
                    })
                    chunk_num += 1
                    current = []
                    current_len = 0
            if current:
                sub_text = " ".join(current).strip()
                if len(sub_text) >= 30:
                    d = extract_date(sub_text)
                    chunks.append({
                        "title": source_name,
                        "category": f"Uploaded • {source_name}",
                        "content": sub_text,
                        "id": f"{source_name}-{chunk_num}",
                        "detected_date": d.iso_date if d else "",
                        "detected_date_str": d.display_str if d else "",
                        "publisher": source_name,
                        "url": ""
                    })
                    chunk_num += 1
        else:
            d = extract_date(norm_block)
            chunks.append({
                "title": source_name,
                "category": f"Uploaded • {source_name}",
                "content": norm_block,
                "id": f"{source_name}-{chunk_num}",
                "detected_date": d.iso_date if d else "",
                "detected_date_str": d.display_str if d else "",
                "publisher": source_name,
                "url": ""
            })
            chunk_num += 1

    return chunks
