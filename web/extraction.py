"""
web/extraction.py
Robust webpage and article extraction using BeautifulSoup.
Strips boilerplate, navigation, ads, and cookies while preserving
article body, publication date, title, and publisher.
Includes in-memory TTL caching and graceful fallback.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment

from web.sources import clean_url, get_domain, get_source_quality

# Simple in-memory cache: url -> (timestamp, ExtractedArticle)
_ARTICLE_CACHE: Dict[str, Tuple[float, "ExtractedArticle"]] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


@dataclass
class ExtractedArticle:
    url: str
    title: str
    publisher: str
    pub_date: str                 # YYYY-MM-DD or display string
    body: str                     # Cleaned article body text
    paragraphs: List[str]         # Filtered readable paragraphs
    quality_score: float
    tier: int
    extracted_successfully: bool
    fallback_used: bool = False


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 InfoQuery/2.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

UNWANTED_TAGS = [
    "script", "style", "nav", "footer", "header", "aside", "noscript",
    "form", "svg", "iframe", "button", "dialog", "canvas"
]

BOILERPLATE_PATTERNS = re.compile(
    r"(subscribe|cookie policy|privacy policy|terms of service|all rights reserved|"
    r"sign in|sign up|advertisement|sponsored|follow us on|download the app|"
    r"newsletter|read more|recommended stories|click here|copyright \d{4})",
    re.IGNORECASE
)


def _clean_paragraph(p_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", p_text or "").strip()
    return cleaned


def extract_date_from_html(soup: BeautifulSoup, text: str) -> str:
    """Extract publication date from meta tags or text."""
    # 1. Meta tags
    meta_props = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "publication_date"}),
        ("meta", {"name": "date"}),
        ("meta", {"name": "pubdate"}),
        ("meta", {"itemprop": "datePublished"}),
        ("meta", {"property": "og:published_time"}),
    ]
    for tag_name, attrs in meta_props:
        el = soup.find(tag_name, attrs)
        if el and el.get("content"):
            content = el["content"][:10]  # grab YYYY-MM-DD
            if re.match(r"^\d{4}-\d{2}-\d{2}", content):
                return content

    # 2. Time tags
    time_tag = soup.find("time")
    if time_tag:
        dt = time_tag.get("datetime") or time_tag.get_text()
        if dt:
            m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", dt)
            if m:
                return m.group(1)

    # 3. Regex on page text
    m = re.search(r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", text[:2000])
    if m:
        return m.group(1)

    return ""


def extract_article_content(
    url: str,
    default_title: str = "",
    default_publisher: str = "",
    default_pub_date: str = "",
    fallback_snippet: str = "",
    timeout: int = 4
) -> ExtractedArticle:
    """
    Fetch article webpage, parse with BeautifulSoup, clean boilerplate,
    and return structured ExtractedArticle. Gracefully falls back to snippet.
    """
    cleaned_url = clean_url(url)
    quality_score, tier, _ = get_source_quality(cleaned_url, default_publisher)

    # Check cache
    now = time.time()
    if cleaned_url in _ARTICLE_CACHE:
        ts, cached_article = _ARTICLE_CACHE[cleaned_url]
        if now - ts < CACHE_TTL_SECONDS:
            return cached_article

    # Wikipedia extract shortcut: if Wikipedia API URL, return directly
    if "wikipedia.org" in cleaned_url and fallback_snippet:
        art = ExtractedArticle(
            url=cleaned_url,
            title=default_title or "Wikipedia Reference",
            publisher=default_publisher or "Wikipedia",
            pub_date=default_pub_date or "",
            body=fallback_snippet,
            paragraphs=[fallback_snippet],
            quality_score=quality_score,
            tier=tier,
            extracted_successfully=True,
            fallback_used=False
        )
        _ARTICLE_CACHE[cleaned_url] = (now, art)
        return art

    # Attempt HTTP fetch
    try:
        response = requests.get(cleaned_url, headers=DEFAULT_HEADERS, timeout=timeout)
        if response.status_code != 200 or not response.text:
            raise ValueError(f"HTTP status {response.status_code}")

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove unwanted tags
        for tag_name in UNWANTED_TAGS:
            for el in soup.find_all(tag_name):
                el.decompose()

        # Remove elements with ad/nav classes or IDs
        for el in soup.find_all(attrs={"class": re.compile(r"(cookie|advert|banner|popup|social-share|newsletter|footer-links)", re.I)}):
            el.decompose()

        # Extract title
        extracted_title = default_title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            candidate_title = og_title["content"].strip()
            if candidate_title.lower() not in ["google news", "google", "news", "home", "latest news"]:
                extracted_title = candidate_title
        elif soup.find("h1"):
            h1 = soup.find("h1").get_text().strip()
            if len(h1) > 5 and h1.lower() not in ["google news", "google", "news"]:
                extracted_title = h1

        # Extract publisher
        extracted_publisher = default_publisher
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            candidate_pub = og_site["content"].strip()
            if candidate_pub.lower() not in ["google news", "google"]:
                extracted_publisher = candidate_pub
        if not extracted_publisher or extracted_publisher.lower() in ["google news", "google"]:
            domain = get_domain(cleaned_url)
            if domain and "google" not in domain:
                extracted_publisher = domain.split(".")[0].capitalize()
            else:
                extracted_publisher = default_publisher or "News Source"

        # Extract pub date
        page_text = soup.get_text()
        extracted_date = default_pub_date or extract_date_from_html(soup, page_text)

        # Extract body text from paragraphs
        paragraphs = []
        # Prefer <article> or [role="main"] if present
        main_container = soup.find("article") or soup.find(attrs={"role": "main"}) or soup

        for p in main_container.find_all("p"):
            p_text = _clean_paragraph(p.get_text())
            # Filter short fragments and boilerplate
            if len(p_text) >= 40:
                if not BOILERPLATE_PATTERNS.search(p_text):
                    paragraphs.append(p_text)

        # If too few paragraphs extracted, fallback
        if len(paragraphs) < 2 and len(" ".join(paragraphs)) < 120:
            if fallback_snippet:
                paragraphs = [fallback_snippet]
                body_text = fallback_snippet
                fallback_used = True
                success = False
            else:
                body_text = " ".join(paragraphs) if paragraphs else default_title
                fallback_used = False
                success = len(body_text) > 30
        else:
            body_text = "\n\n".join(paragraphs)
            fallback_used = False
            success = True

        article = ExtractedArticle(
            url=cleaned_url,
            title=extracted_title or default_title,
            publisher=extracted_publisher or default_publisher,
            pub_date=extracted_date,
            body=body_text,
            paragraphs=paragraphs,
            quality_score=quality_score,
            tier=tier,
            extracted_successfully=success,
            fallback_used=fallback_used
        )
        _ARTICLE_CACHE[cleaned_url] = (now, article)
        return article

    except Exception:
        # Graceful fallback to snippet
        body_text = fallback_snippet or default_title
        paragraphs = [body_text] if body_text else []
        article = ExtractedArticle(
            url=cleaned_url,
            title=default_title,
            publisher=default_publisher,
            pub_date=default_pub_date,
            body=body_text,
            paragraphs=paragraphs,
            quality_score=quality_score,
            tier=tier,
            extracted_successfully=False,
            fallback_used=True
        )
        _ARTICLE_CACHE[cleaned_url] = (now, article)
        return article


def batch_extract_articles(
    items: List[Dict],
    max_workers: int = 5,
    timeout: int = 4
) -> List[ExtractedArticle]:
    """
    Concurrently extract articles for a list of candidate search items.
    Each item dict should have: url, title, publisher, pub_date, snippet.
    """
    articles: List[ExtractedArticle] = []
    if not items:
        return articles

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                extract_article_content,
                item.get("url", ""),
                item.get("title", ""),
                item.get("publisher", ""),
                item.get("pub_date", ""),
                item.get("snippet", ""),
                timeout
            ): item
            for item in items
        }
        for future in as_completed(future_map):
            try:
                res = future.result()
                articles.append(res)
            except Exception:
                pass

    return articles
