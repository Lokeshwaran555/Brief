"""Map an evidence item's source/url to a typed category.

Categories drive the drawer's source-mix breakdown ("3 portal · 2 social
· 1 official_dev") and let the implication writer weight evidence by
type when drawing conclusions.
"""
from __future__ import annotations

from urllib.parse import urlparse


OFFICIAL_DEV_DOMAINS = {
    "aldar.com", "emaar.com", "damacproperties.com", "binghatti.com",
    "binghatti-dubai.com", "sobharealty.com", "nakheel.com",
    "ellingtonproperties.ae", "azizidevelopments.com", "omniyat.com",
    "deyaar.ae", "modon.ae", "meraas.com", "dubaiholding.com",
}
PORTAL_DOMAINS = {
    "bayut.com", "propertyfinder.ae", "dubizzle.com",
    "houza.ae", "haus.ae",
}
NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "khaleejtimes.com", "thenationalnews.com", "gulfnews.com",
    "arabianbusiness.com", "zawya.com", "wam.ae",
    "news.google.com",
}
REGULATORY_DOMAINS = {
    "dld.gov.ae", "rera.gov.ae", "land.gov.ae",
    "ded.ae", "fatf-gafi.org",
}


def _host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _domain_matches(host: str, candidates: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in candidates)


def classify(source: str | None, url: str | None) -> str:
    """Returns one of:
    official_dev | portal | broker | social | video | forum | news |
    regulatory | search | internal
    """
    s = (source or "").lower()
    host = _host(url)

    # Internal — our own signals table
    if s.startswith("signals"):
        return "internal"

    # Official developer sites
    if _domain_matches(host, OFFICIAL_DEV_DOMAINS):
        return "official_dev"

    # Listing portals
    if _domain_matches(host, PORTAL_DOMAINS):
        return "portal"
    if s.startswith("bayut") or s.startswith("dubizzle"):
        return "portal"
    if s.startswith("forum:pf-blog") or s.startswith("forum:bayut-blog"):
        return "portal"

    # Video
    if s.startswith("youtube") or "youtube.com" in host or host == "youtu.be":
        return "video"

    # Social
    if s.startswith("instagram") or "instagram.com" in host:
        return "social"
    if s.startswith("linkedin") or "linkedin.com" in host:
        return "social"

    # Forum
    if s.startswith("reddit") or "reddit.com" in host:
        return "forum"
    if s.startswith("forum:expat") or "expatforum.com" in host:
        return "forum"

    # Regulatory
    if _domain_matches(host, REGULATORY_DOMAINS):
        return "regulatory"

    # News (mainstream wire)
    if _domain_matches(host, NEWS_DOMAINS):
        return "news"
    if s.startswith("gnews"):
        return "news"

    # Generic search (Tavily catches everything else)
    return "search"


# Reliability weights — useful for downstream LLM context
RELIABILITY_BY_TYPE: dict[str, float] = {
    "official_dev": 1.0,
    "regulatory":   1.0,
    "news":         0.85,
    "portal":       0.75,
    "video":        0.65,
    "social":       0.55,
    "forum":        0.40,
    "search":       0.50,
    "internal":     0.70,
    "broker":       0.65,
}


def reliability(evidence_type: str | None) -> float:
    return RELIABILITY_BY_TYPE.get((evidence_type or "search").lower(), 0.5)


# Evidence-class buckets for the strength rule below.
_OFFICIAL_LIKE = {"official_dev", "regulatory", "internal"}
_NEWS_LIKE = {"news", "portal"}
_SOCIAL_LIKE = {"social", "video", "broker", "forum"}


def evidence_strength(evidence: list[dict[str, str]]) -> str:
    """Classify a full evidence pack into one of four bands.

    Rule (matches the dashboard's Strong/Medium/Weak/Conflicted contract):
      - Strong:   ≥1 official-like + ≥1 news-like + ≥1 social/video/broker
                  (multi-class corroboration with a high-reliability anchor)
      - Medium:   ≥1 news-like + ≥1 social/video/broker
                  (corroboration without an official anchor)
      - Conflicted: any 'unresolved' counts surfaced + multi-class evidence
                  (handled at caller-level since this fn doesn't see facts)
      - Weak:     anything else (search/forum-only or single-class)
    """
    if not evidence:
        return "weak"
    types_seen: set[str] = set()
    for e in evidence:
        t = (e.get("evidence_type") or "search").lower()
        types_seen.add(t)
    has_official = bool(types_seen & _OFFICIAL_LIKE)
    has_news = bool(types_seen & _NEWS_LIKE)
    has_social = bool(types_seen & _SOCIAL_LIKE)
    if has_official and has_news and has_social:
        return "strong"
    if has_news and has_social:
        return "medium"
    if has_official and (has_news or has_social):
        return "medium"
    return "weak"


def aggregate_reliability(evidence: list[dict[str, str]]) -> float:
    """Mean reliability across the pack. 0-1. Useful as a numeric
    companion to the band — lets the dashboard show 'Medium · 72%'.
    """
    if not evidence:
        return 0.0
    scores = [reliability(e.get("evidence_type")) for e in evidence]
    return sum(scores) / len(scores)
