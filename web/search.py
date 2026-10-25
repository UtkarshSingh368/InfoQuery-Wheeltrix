"""
web/search.py
Multi-source web and news search engine.
Discovers candidates via Google News RSS, direct publisher feeds (BBC, NDTV,
The Hindu, Times of India, Yahoo), and Wikipedia API.
Deduplicates URLs and cluster-redundant reports, extracts articles,
and chunks them into searchable passages.
"""

from __future__ import annotations

import difflib
import email.utils
import html
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import requests

from core.query_understanding import QueryAnalysis, TargetDate, normalize_text, tokenize
from web.extraction import ExtractedArticle, batch_extract_articles, extract_article_content
from web.sources import clean_url, get_domain, get_source_quality

# Search cache: (query_str, intent) -> (timestamp, List[Dict])
_SEARCH_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}
SEARCH_CACHE_TTL = 600  # 10 minutes

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 InfoQuery-Search/2.0"
    )
}

DIRECT_RSS_FEEDS = [
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "category": "World News"},
    {"name": "BBC Tech", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "category": "Technology"},
    {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport/rss.xml", "category": "Sports"},
    {"name": "NDTV", "url": "https://feeds.feedburner.com/ndtvnews-top-stories", "category": "India News"},
    {"name": "The Hindu", "url": "https://www.thehindu.com/news/national/feeder/default.rss", "category": "India News"},
    {"name": "Times of India", "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "category": "India News"},
    {"name": "Yahoo News", "url": "https://news.yahoo.com/rss/", "category": "General News"},
]


def _parse_rfc822_date(date_str: str) -> Tuple[str, Optional[datetime]]:
    """Parse RSS pubDate like 'Sun, 25 Oct 2026 05:28:14 GMT' to 'YYYY-MM-DD'."""
    if not date_str:
        return "", None
    try:
        parsed_tuple = email.utils.parsedate_to_datetime(date_str)
        iso = parsed_tuple.strftime("%Y-%m-%d")
        return iso, parsed_tuple
    except Exception:
        # Fallback regex
        m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", date_str)
        if m:
            return f"{m.group(1)} {m.group(2)} {m.group(3)}", None
        return date_str[:10], None


def search_google_news_rss(query: str, max_items: int = 6, timeout: int = 3) -> List[Dict]:
    """Search Google News RSS for a specific query."""
    candidates = []
    # Pick optimal region parameters based on query context
    q_low = query.lower()
    if "india" in q_low or "delhi" in q_low or "cricket" in q_low:
        gl, hl, ceid = "IN", "en-IN", "IN:en"
    else:
        gl, hl, ceid = "US", "en-US", "US:en"

    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl={hl}&gl={gl}&ceid={ceid}"
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            items = root.findall("./channel/item")
            for item in items[:max_items]:
                raw_title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub_date_raw = item.findtext("pubDate") or ""
                desc = item.findtext("description") or ""

                source_el = item.find("source")
                publisher = source_el.text if source_el is not None else ""
                source_url = source_el.get("url", "") if source_el is not None else ""

                # Extract title without trailing ' - Publisher'
                clean_title = raw_title
                if " - " in raw_title:
                    clean_title, trailing_pub = raw_title.rsplit(" - ", 1)
                    if not publisher:
                        publisher = trailing_pub.strip()

                iso_date, _ = _parse_rfc822_date(pub_date_raw)
                clean_desc = normalize_text(re.sub(r"<[^>]+>", " ", desc))

                candidates.append({
                    "title": clean_title.strip(),
                    "url": link,
                    "publisher": publisher or "Google News",
                    "pub_date": iso_date,
                    "raw_date": pub_date_raw,
                    "snippet": clean_desc,
                    "source_type": "Google News RSS",
                    "source_url": source_url,
                })
    except Exception:
        pass
    return candidates


def search_wikipedia_api(query: str, max_items: int = 3, timeout: int = 5) -> List[Dict]:
    """Search Wikipedia for encyclopedic/factual topics."""
    candidates = []
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_items,
            "format": "json",
            "utf8": 1,
        }
        resp = requests.get(search_url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            for res in results:
                title = res.get("title", "")
                snippet = normalize_text(re.sub(r"<[^>]+>", " ", res.get("snippet", "")))

                # Fetch clean text extract with redirects followed
                extract_params = {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exintro": 1,
                    "redirects": 1,
                    "titles": title,
                    "format": "json",
                    "utf8": 1,
                }
                ep_resp = requests.get(search_url, params=extract_params, headers=DEFAULT_HEADERS, timeout=timeout)
                extract_text = ""
                if ep_resp.status_code == 200:
                    pages = ep_resp.json().get("query", {}).get("pages", {})
                    if pages:
                        extract_text = next(iter(pages.values())).get("extract", "")

                body = normalize_text(extract_text or snippet)
                if body:
                    candidates.append({
                        "title": title,
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "publisher": "Wikipedia",
                        "pub_date": "",
                        "snippet": body[:500],
                        "full_extract": body,
                        "source_type": "Wikipedia API"
                    })
    except Exception:
        pass
    return candidates


def search_direct_rss_feeds(keywords: List[str], timeout: int = 4) -> List[Dict]:
    """Search curated high-reputation direct RSS feeds for query keywords."""
    candidates = []
    kw_set = {k.lower() for k in keywords if len(k) > 2}
    if not kw_set:
        return candidates

    for feed in DIRECT_RSS_FEEDS:
        try:
            resp = requests.get(feed["url"], headers=DEFAULT_HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall("./channel/item")[:15]:
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub_date = item.findtext("pubDate") or ""
                desc = item.findtext("description") or ""

                text_to_check = (title + " " + desc).lower()
                # Check keyword overlap
                matches = sum(1 for kw in kw_set if kw in text_to_check)
                if matches >= 1:
                    iso_date, _ = _parse_rfc822_date(pub_date)
                    clean_desc = normalize_text(re.sub(r"<[^>]+>", " ", desc))
                    candidates.append({
                        "title": title.strip(),
                        "url": link,
                        "publisher": feed["name"],
                        "pub_date": iso_date,
                        "raw_date": pub_date,
                        "snippet": clean_desc,
                        "source_type": "Direct RSS Feed",
                        "category": feed["category"]
                    })
        except Exception:
            continue
    return candidates


def deduplicate_and_cluster(candidates: List[Dict]) -> List[Dict]:
    """
    Remove duplicate URLs and cluster articles covering the identical event.
    Keeps the higher quality source or more informative version.
    """
    deduped = []
    seen_urls = set()

    for item in candidates:
        raw_url = item.get("url", "")
        cleaned = clean_url(raw_url)
        if not cleaned:
            continue
        if cleaned in seen_urls:
            continue

        # Check title similarity with already selected items
        title = item.get("title", "").lower().strip()
        is_duplicate = False
        for existing in deduped:
            existing_title = existing.get("title", "").lower().strip()
            # If same publisher and same date and high title similarity
            ratio = difflib.SequenceMatcher(None, title, existing_title).ratio()
            if ratio > 0.78:
                # Prioritize higher tier source
                score_item, _, _ = get_source_quality(cleaned, item.get("publisher", ""))
                score_ex, _, _ = get_source_quality(existing.get("url", ""), existing.get("publisher", ""))
                if score_item > score_ex:
                    existing.update(item)
                is_duplicate = True
                break

        if not is_duplicate:
            seen_urls.add(cleaned)
            deduped.append(item)

    return deduped


def passage_chunk_article(article: ExtractedArticle, target_date: Optional[TargetDate] = None) -> List[Dict]:
    """
    Split an extracted article into searchable passages with rich metadata.
    Preserves title, publisher, URL, publication date, and quality tier.
    """
    chunks = []
    paragraphs = article.paragraphs
    if not paragraphs:
        paragraphs = [article.body] if article.body else []

    # Window paragraphs into chunks of 1 to 2 paragraphs (~60-150 words)
    i = 0
    chunk_idx = 1
    while i < len(paragraphs):
        p1 = paragraphs[i].strip()
        p2 = paragraphs[i + 1].strip() if i + 1 < len(paragraphs) else ""

        # If p1 is substantial, it can be its own chunk
        if len(p1) > 200 or not p2:
            combined = p1
            i += 1
        else:
            combined = f"{p1} {p2}"
            i += 2

        if len(combined) < 35:
            continue

        chunks.append({
            "title": article.title,
            "content": combined,
            "url": article.url,
            "publisher": article.publisher,
            "pub_date": article.pub_date,
            "tier": article.tier,
            "quality_score": article.quality_score,
            "category": f"Live Web • {article.publisher}",
            "id": f"{article.publisher}-{chunk_idx}",
            "fallback_used": article.fallback_used
        })
        chunk_idx += 1

    return chunks


def perform_live_web_search(
    question: str,
    analysis: QueryAnalysis,
    max_candidates: int = 8,
    max_passages: int = 25
) -> List[Dict]:
    """
    Execute end-to-end Live Web Search pipeline:
      1. Run focused multi-angle queries
      2. Discover candidates across Google News, Direct Feeds, and Wikipedia
      3. Deduplicate and cluster similar stories
      4. Extract article content in parallel
      5. Chunk articles into rich passages
    """
    cache_key = f"{question.strip().lower()}||{analysis.intent}"
    now = time.time()
    if cache_key in _SEARCH_CACHE:
        ts, cached = _SEARCH_CACHE[cache_key]
        if now - ts < SEARCH_CACHE_TTL:
            return cached

    discovered: List[Dict] = []

    # Parallelize multi-query discovery across sources
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        # Google News RSS for generated queries
        for q_term in analysis.search_queries[:4]:
            futures.append(executor.submit(search_google_news_rss, q_term, 5, 3))

        # Wikipedia for factual, definitions, tech, science, or background
        if analysis.intent in ["definition", "technology", "science", "sports", "factual", "person", "general"]:
            futures.append(executor.submit(search_wikipedia_api, " ".join(analysis.keywords[:4]), 2, 3))
            if analysis.entities:
                futures.append(executor.submit(search_wikipedia_api, analysis.entities[0], 1, 3))

        # Direct RSS feeds for breaking/current news
        if analysis.is_latest or analysis.intent in ["daily_news_summary", "conflict", "politics"]:
            futures.append(executor.submit(search_direct_rss_feeds, analysis.keywords[:4], 3))

        for f in as_completed(futures):
            try:
                res = f.result()
                if res:
                    discovered.extend(res)
            except Exception:
                pass

    if not discovered:
        # Fallback query with raw question
        discovered.extend(search_google_news_rss(question, max_items=5, timeout=3))

    # 4. Deduplicate and cluster
    unique_candidates = deduplicate_and_cluster(discovered)

    # Prioritize candidates based on quality score and date alignment
    def rank_candidate(c: Dict) -> float:
        score, _, _ = get_source_quality(c.get("url", ""), c.get("publisher", ""))
        # Date match bonus
        c_date = c.get("pub_date", "")
        if analysis.target_date and c_date:
            if analysis.target_date.iso_date in c_date or analysis.target_date.display_str in c_date:
                score += 0.50
        return score

    unique_candidates.sort(key=rank_candidate, reverse=True)
    top_candidates = unique_candidates[:max_candidates]

    # 5. Extract article contents in parallel
    extracted_articles = batch_extract_articles(top_candidates, max_workers=5, timeout=4)

    # 6. Chunk into passages
    all_passages: List[Dict] = []
    for art in extracted_articles:
        passages = passage_chunk_article(art, analysis.target_date)
        all_passages.extend(passages)

    # If no passages could be chunked, use snippets as fallback records
    if not all_passages:
        for c in top_candidates:
            snippet = c.get("snippet", "") or c.get("title", "")
            if snippet:
                all_passages.append({
                    "title": c.get("title", "Web Result"),
                    "content": snippet,
                    "url": c.get("url", ""),
                    "publisher": c.get("publisher", "Web"),
                    "pub_date": c.get("pub_date", ""),
                    "tier": 4,
                    "quality_score": 0.40,
                    "category": f"Live Web • {c.get('publisher', 'Snippet')}",
                    "id": f"web-{len(all_passages)+1}",
                    "fallback_used": True
                })

    final_passages = all_passages[:max_passages]
    _SEARCH_CACHE[cache_key] = (now, final_passages)
    return final_passages
