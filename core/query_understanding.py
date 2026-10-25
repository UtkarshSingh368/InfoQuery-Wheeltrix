"""
core/query_understanding.py
Analyzes user questions to extract intents, target dates, entities,
and generate focused multi-angle search queries.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "give", "show", "know"
}


@dataclass
class TargetDate:
    iso_date: str                 # YYYY-MM-DD
    day: int
    month: int
    year: int
    display_str: str              # e.g. "25 October 2026"
    month_name: str               # "October"
    is_today: bool = False


@dataclass
class QueryAnalysis:
    raw_question: str
    clean_question: str
    intent: str                   # daily_news_summary, latest_news, definition, sports, etc.
    target_date: Optional[TargetDate]
    is_latest: bool
    entities: List[str]
    keywords: List[str]
    search_queries: List[str]
    category_hint: Optional[str] = None


def normalize_text(text: str) -> str:
    """Strip whitespace, unescape HTML entities, and collapse multiple spaces."""
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric word tokens."""
    return re.findall(r"(?u)\b\w+\b", (text or "").lower())


def extract_date(text: str) -> Optional[TargetDate]:
    """
    Extract explicit or relative dates from text.
    Handles formats like:
      - '25 October 2026', '25th October 2026'
      - 'October 25, 2026', 'October 25 2026'
      - '2026-10-25', '25-10-2026'
    """
    t = text.lower()

    # Pattern 1: Day Month Year (e.g. 25th October 2026 or 25 October 2026)
    m1 = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})\b",
        text,
        re.IGNORECASE
    )
    if m1:
        d, m_name, y = int(m1.group(1)), m1.group(2).lower(), int(m1.group(3))
        if m_name in MONTHS and 1 <= d <= 31:
            m_num = MONTHS[m_name]
            month_title = m1.group(2).capitalize()
            return TargetDate(
                iso_date=f"{y:04d}-{m_num:02d}-{d:02d}",
                day=d,
                month=m_num,
                year=y,
                display_str=f"{d} {month_title} {y}",
                month_name=month_title
            )

    # Pattern 2: Month Day, Year (e.g. October 25, 2026)
    m2 = re.search(
        r"\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b",
        text,
        re.IGNORECASE
    )
    if m2:
        m_name, d, y = m2.group(1).lower(), int(m2.group(2)), int(m2.group(3))
        if m_name in MONTHS and 1 <= d <= 31:
            m_num = MONTHS[m_name]
            month_title = m2.group(1).capitalize()
            return TargetDate(
                iso_date=f"{y:04d}-{m_num:02d}-{d:02d}",
                day=d,
                month=m_num,
                year=y,
                display_str=f"{d} {month_title} {y}",
                month_name=month_title
            )

    # Pattern 3: YYYY-MM-DD
    m3 = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m3:
        y, m, d = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            month_names = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            m_name = month_names[m - 1]
            return TargetDate(
                iso_date=f"{y:04d}-{m:02d}-{d:02d}",
                day=d,
                month=m,
                year=y,
                display_str=f"{d} {m_name} {y}",
                month_name=m_name
            )

    # Pattern 4: DD/MM/YYYY or DD-MM-YYYY
    m4 = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text)
    if m4:
        d, m, y = int(m4.group(1)), int(m4.group(2)), int(m4.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            month_names = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            m_name = month_names[m - 1]
            return TargetDate(
                iso_date=f"{y:04d}-{m:02d}-{d:02d}",
                day=d,
                month=m,
                year=y,
                display_str=f"{d} {m_name} {y}",
                month_name=m_name
            )

    # Pattern 5: Relative "today" or "recent"
    if "today" in t or "right now" in t:
        now = datetime.now(timezone.utc)
        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        m_name = month_names[now.month - 1]
        return TargetDate(
            iso_date=f"{now.year:04d}-{now.month:02d}-{now.day:02d}",
            day=now.day,
            month=now.month,
            year=now.year,
            display_str=f"{now.day} {m_name} {now.year}",
            month_name=m_name,
            is_today=True
        )

    return None


def extract_entities(text: str) -> List[str]:
    """
    Extract key named entities and capitalized sequences
    (e.g. 'FIFA World Cup 2026', 'T20 World Cup', 'Ukraine', 'Spain').
    """
    entities = []
    # Match sequences of Title Case words or uppercase abbreviations
    matches = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text)
    for m in matches:
        m_clean = m.strip()
        tokens = m_clean.split()
        # Avoid standalone single stopword like 'What', 'Who', 'When'
        if len(tokens) == 1 and tokens[0].lower() in STOP_WORDS:
            continue
        if len(m_clean) >= 2 and m_clean not in entities:
            entities.append(m_clean)

    # Specific well-known acronyms or terms if present
    common_terms = ["FIFA", "ICC", "T20", "BM25", "TF-IDF", "AI", "ML", "GPS", "HTTP"]
    for term in common_terms:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) and term not in entities:
            entities.append(term)

    return entities


def detect_intent(question: str, target_date: Optional[TargetDate]) -> str:
    """
    Classify user question into focused intents.
    """
    q = question.lower().strip()

    # 1. Daily news summary
    if target_date and not target_date.is_today:
        if re.search(r"\b(what happened|news|developments|headlines|events|summary)\b", q):
            return "daily_news_summary"
        if re.match(r"^(what|tell me|any news)\b", q):
            return "daily_news_summary"

    # 2. Latest news
    if any(k in q for k in ["latest news", "recent news", "breaking news", "news about", "updates on", "headlines", "what is happening"]):
        return "latest_news"

    # 3. Sports
    if any(k in q for k in ["world cup", "fifa", "icc", "t20", "cricket", "football", "soccer", "tennis", "basketball", "olympic", "match", "tournament", "score", "won", "winner", "champion"]):
        return "sports"

    # 4. Conflict / War
    if any(k in q for k in ["ukraine", "russia", "gaza", "israel", "war", "conflict", "drone", "missile", "strike", "attack", "invasion", "military"]):
        return "conflict"

    # 5. Politics
    if any(k in q for k in ["election", "parliament", "congress", "president", "prime minister", "government", "policy", "minister", "senate"]):
        return "politics"

    # 6. Definition
    if re.match(r"^(what is|what are|define|explain|meaning of|what do you mean by)\b", q):
        return "definition"

    # 7. Person / Biography
    if re.match(r"^(who is|who was|who were)\b", q):
        return "person"

    # 8. Technology
    if any(k in q for k in ["machine learning", "artificial intelligence", "bm25", "tf-idf", "algorithm", "cloud computing", "database", "neural network", "deep learning"]):
        return "technology"

    # 9. Science
    if any(k in q for k in ["photosynthesis", "water cycle", "climate change", "biology", "physics", "chemistry"]):
        return "science"

    # 10. General fact
    if re.match(r"^(who|which|where|when|why|how)\b", q):
        return "factual"

    return "general"


def extract_keywords(question: str) -> List[str]:
    """Extract significant content words from query."""
    tokens = tokenize(question)
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


def generate_search_queries(
    question: str,
    intent: str,
    target_date: Optional[TargetDate],
    entities: List[str],
    keywords: List[str]
) -> List[str]:
    """
    Generate multiple focused, high-precision search queries.
    Never relies on just one generic query.
    """
    queries = []
    clean_q = normalize_text(question)

    if intent == "daily_news_summary" and target_date:
        d_str = target_date.display_str  # e.g. "25 October 2026"
        iso_str = target_date.iso_date   # e.g. "2026-10-25"
        alt_str = f"{target_date.month_name} {target_date.day} {target_date.year}"

        # Generate multi-category news queries
        queries.append(f"{d_str} major news India")
        queries.append(f"{d_str} world news headlines")
        queries.append(f"{d_str} politics developments")
        queries.append(f"{d_str} sports news")
        queries.append(f"{d_str} conflict war news")
        queries.append(f"{alt_str} news")
        queries.append(f"news on {d_str}")

    elif intent == "latest_news":
        # Extract main subject
        topic_words = [k for k in keywords if k not in ["latest", "recent", "news", "today", "update", "updates"]]
        topic = " ".join(topic_words) if topic_words else "world"
        queries.append(f"{topic} latest news updates")
        queries.append(f"{topic} breaking news today")
        queries.append(f"{topic} conflict developments")
        queries.append(f"{topic} current news")

    elif intent == "sports":
        # For sports, include entity, tournament, and winner
        ent_str = " ".join(entities) if entities else " ".join(keywords)
        queries.append(f"{ent_str} winner final champion")
        queries.append(f"{ent_str} match results")
        queries.append(f"{clean_q}")

    elif intent == "definition":
        core = " ".join([k for k in keywords if k not in ["what", "define", "explain", "meaning"]])
        queries.append(f"{core} definition explanation overview")
        queries.append(f"{core} artificial intelligence" if "learning" in core else f"{core} concept")
        queries.append(clean_q)

    elif intent == "conflict":
        topic = " ".join(entities) or " ".join(keywords)
        queries.append(f"{topic} war conflict latest news")
        queries.append(f"{topic} military updates")
        queries.append(clean_q)

    else:
        # Default multi-angle search
        if entities:
            queries.append(" ".join(entities))
        if keywords:
            queries.append(" ".join(keywords[:5]))
        queries.append(clean_q)

    # Deduplicate while preserving order
    seen = set()
    final_queries = []
    for q in queries:
        q_norm = normalize_text(q).lower()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            final_queries.append(q)

    return final_queries[:6]


def analyze_query(question: str) -> QueryAnalysis:
    """
    Main entry point for Query Understanding.
    Takes user natural language question and returns comprehensive structured analysis.
    """
    clean_q = normalize_text(question)
    target_date = extract_date(clean_q)
    intent = detect_intent(clean_q, target_date)
    is_latest = intent in ["latest_news"] or (target_date is not None and target_date.is_today) or "latest" in clean_q.lower()
    entities = extract_entities(clean_q)
    keywords = extract_keywords(clean_q)
    search_queries = generate_search_queries(clean_q, intent, target_date, entities, keywords)

    return QueryAnalysis(
        raw_question=question,
        clean_question=clean_q,
        intent=intent,
        target_date=target_date,
        is_latest=is_latest,
        entities=entities,
        keywords=keywords,
        search_queries=search_queries,
    )
