"""Geopolitics-with-RE-impact scout — only the geopolitics that actually matter.

Stakeholder ask (2026-04-28): "Geopolitics that actually matter."
Filter geopolitics through a real-estate lens: visa rule changes,
capital controls, sanctions affecting cross-border flows, supply-chain
shocks (Red Sea, Suez), regional conflict spillover that moves
HNWI migration into / out of Dubai.

Strategy: focused Tavily queries pairing each geopolitical theme
with RE-impact terms (visa, capital flow, sanctions, supply chain,
real estate, property). Reuters / FT / Bloomberg / Al Jazeera /
Reuters / The National / Khaleej Times are the trusted sources.

This complements gdelt_scout (broad global news) by surfacing only
the geopolitical items with concrete RE consequences.

Free under the existing Tavily plan. ~7 queries per run.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from tools import tavily_search

log = logging.getLogger(__name__)


# Each query pairs a geopolitical theme with RE-impact filter terms.
# Domains are trusted news sources — no GDELT noise here, this is the
# curated complement to the broad GDELT firehose.
GEOPOLITICS_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "Israel-Iran-Houthi Red Sea",
        "query":   "Israel Iran Houthi Red Sea shipping disruption real estate construction supply chain Dubai",
        "domains": ["reuters.com", "ft.com", "bloomberg.com", "aljazeera.com",
                    "thenationalnews.com", "khaleejtimes.com", "gulfnews.com"],
        "country_code": None,
        "topic":   "MENA conflict / shipping",
    },
    {
        "label":   "Russia sanctions capital flow",
        "query":   "Russia sanctions UAE Dubai property capital flight HNWI 2026",
        "domains": ["reuters.com", "ft.com", "bloomberg.com", "thenationalnews.com",
                    "khaleejtimes.com", "gulfnews.com", "arabianbusiness.com"],
        "country_code": "RU",
        "topic":   "Russia / sanctions",
    },
    {
        "label":   "China outbound capital",
        "query":   "China outbound capital flow real estate UAE Dubai 2026 HNWI investment",
        "domains": ["reuters.com", "ft.com", "bloomberg.com", "scmp.com",
                    "thenationalnews.com", "khaleejtimes.com"],
        "country_code": "CN",
        "topic":   "China / capital outflow",
    },
    {
        "label":   "US-China tariffs",
        "query":   "US China tariffs trade real estate construction supply chain 2026",
        "domains": ["reuters.com", "ft.com", "bloomberg.com", "wsj.com",
                    "thenationalnews.com", "scmp.com"],
        "country_code": None,
        "topic":   "US-China trade",
    },
    {
        "label":   "India RBI LRS overseas property",
        "query":   "India RBI LRS overseas property purchase Dubai rule change 2026",
        "domains": ["reuters.com", "ft.com", "bloomberg.com",
                    "economictimes.indiatimes.com", "livemint.com",
                    "thenationalnews.com", "khaleejtimes.com"],
        "country_code": "IN",
        "topic":   "India / RBI policy",
    },
    {
        "label":   "Pakistan stability HNWI outflow",
        "query":   "Pakistan political stability HNWI capital outflow Dubai property 2026",
        "domains": ["reuters.com", "ft.com", "bloomberg.com",
                    "dawn.com", "thenationalnews.com", "khaleejtimes.com"],
        "country_code": "PK",
        "topic":   "Pakistan / HNWI flight",
    },
    {
        "label":   "Saudi-Iran-GCC dynamics",
        "query":   "Saudi Arabia Iran GCC real estate cross-border investment Dubai 2026",
        "domains": ["reuters.com", "ft.com", "bloomberg.com", "aljazeera.com",
                    "thenationalnews.com", "khaleejtimes.com", "gulfnews.com",
                    "arabnews.com"],
        "country_code": None,
        "topic":   "GCC dynamics",
    },
]


MAX_AGE_DAYS = 14


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "geopol"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True
    try:
        s = str(published_str).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


# RE-impact gate: even with a tight query, Tavily occasionally returns
# pure-political stories with no property hook. Drop anything that
# doesn't mention an RE-relevant term in title + content snippet.
RE_IMPACT_TERMS: tuple[str, ...] = (
    "real estate", "property", "housing", "apartment", "villa",
    "developer", "construction", "supply chain", "shipping", "container",
    "capital flow", "capital flight", "hnwi", "high net worth",
    "visa", "residency", "golden visa", "investor visa",
    "mortgage", "rate hike", "investment", "freehold",
    "dubai", "abu dhabi", "uae", "gcc",
)


def _has_re_hook(title: str, content: str) -> bool:
    text = (title + " " + content).lower()
    return any(term in text for term in RE_IMPACT_TERMS)


async def _fetch(query_def: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            query_def["query"],
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=query_def["domains"],
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[geopol:%s] Tavily failed: %s", query_def["label"], e)
        return []

    out: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        content = (r.get("content") or "")
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        if not _has_re_hook(title, content):
            continue
        out.append(
            {
                "source": f"geopolitics:{query_def['label'].lower().replace(' ', '_').replace('/', '_')}",
                "source_url": url,
                "title": f"Geopolitics · {title[:240]}",
                "summary": content[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": "other",  # geopolitics is global; lives on Global RE Pulse page
                    "country_code": query_def.get("country_code"),
                    "category_hint": "geopolitics",
                    "topic": query_def["topic"],
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
    return out


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        k = it["dedup_key"]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


async def run(limit: int = 20) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(q) for q in GEOPOLITICS_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[geopolitics] queries=%d fetched=%d deduped=%d",
        len(GEOPOLITICS_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
