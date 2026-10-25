"""
web/sources.py
Reputation and source-quality scoring based on publisher tiers.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Tuple

TIER_1_PUBLISHERS = {
    "reuters", "associated press", "ap news", "ap", "bbc", "bbc news", "bbc sport",
    "bloomberg", "afp", "official", "who", "un", "united nations",
    "fifa", "icc", "olympics", "nasa", "pib", "government"
}

TIER_1_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com",
    "afp.com", "who.int", "un.org", "fifa.com", "icc-cricket.com",
    "olympics.com", "nasa.gov", "gov.in", "nic.in", "gov", "gov.uk"
}

TIER_2_PUBLISHERS = {
    "the hindu", "the indian express", "indian express", "ndtv", "ndtv news",
    "india today", "times of india", "hindustan times", "cnn", "the guardian",
    "new york times", "the new york times", "washington post", "the washington post",
    "wall street journal", "wsj", "forbes", "espn", "cricbuzz", "cbs news",
    "abc news", "nbc news", "al jazeera", "financial times", "deccan herald",
    "mint", "livemint", "business standard"
}

TIER_2_DOMAINS = {
    "thehindu.com", "indianexpress.com", "ndtv.com", "indiatoday.in",
    "timesofindia.indiatimes.com", "hindustantimes.com", "cnn.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "forbes.com", "espn.com", "espncricinfo.com", "cricbuzz.com",
    "cbsnews.com", "abcnews.go.com", "nbcnews.com", "aljazeera.com",
    "ft.com", "deccanherald.com", "livemint.com", "business-standard.com"
}

TIER_3_DOMAINS = {
    "wikipedia.org", "wikimedia.org", "techcrunch.com", "theverge.com",
    "wired.com", "vogue.in", "elle.com", "ign.com", "medium.com",
    "substack.com", "news.google.com"
}


def clean_url(url: str) -> str:
    """Strip tracking query parameters like utm_*, oc, fbclid, etc."""
    if not url:
        return ""
    # Remove google news wrapper if present in link parameter
    m = re.search(r"[?&]url=([^&]+)", url)
    if m:
        url = m.group(1)
    # Strip tracking parameters
    url = re.sub(r"([?&])(utm_[a-z]+|oc|fbclid|ref|source|at_medium|at_campaign)=[^&]*&?", r"\1", url)
    url = re.sub(r"[?&]$", "", url)
    return url


def get_domain(url: str) -> str:
    """Extract clean domain from URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def get_source_quality(url: str = "", publisher: str = "") -> Tuple[float, int, str]:
    """
    Returns (quality_score, tier, reason).
    Tier 1: 1.0 (Global wire services, official bodies, BBC, Reuters)
    Tier 2: 0.82 (Major established national/international news publications)
    Tier 3: 0.60 (Encyclopedia, tech journals, magazines)
    Tier 4: 0.35 (Unknown, aggregators, blogs)
    """
    pub_lower = (publisher or "").lower().strip()
    domain = get_domain(url)

    # Check Tier 1
    for t1 in TIER_1_PUBLISHERS:
        if re.search(rf"\b{re.escape(t1)}\b", pub_lower):
            return 1.0, 1, f"Tier 1: High-reputation wire service / official body ({publisher})"
    for d1 in TIER_1_DOMAINS:
        if domain == d1 or domain.endswith("." + d1) or domain.endswith(".gov") or domain.endswith(".gov.in"):
            return 1.0, 1, f"Tier 1: Verified official / wire domain ({domain})"

    # Check Tier 2
    for t2 in TIER_2_PUBLISHERS:
        if re.search(rf"\b{re.escape(t2)}\b", pub_lower):
            return 0.85, 2, f"Tier 2: Major established publication ({publisher})"
    for d2 in TIER_2_DOMAINS:
        if domain == d2 or domain.endswith("." + d2):
            return 0.85, 2, f"Tier 2: Major verified news domain ({domain})"

    # Check Tier 3
    if "wikipedia" in pub_lower or "wikipedia.org" in domain:
        return 0.70, 3, "Tier 3: Wikipedia encyclopedia / background"
    for d3 in TIER_3_DOMAINS:
        if domain == d3 or domain.endswith("." + d3):
            return 0.60, 3, f"Tier 3: Magazine / tech blog ({domain})"

    # Fallback Tier 4
    if domain:
        return 0.40, 4, f"Tier 4: General web source ({domain})"
    return 0.35, 4, "Tier 4: Unknown web source"
