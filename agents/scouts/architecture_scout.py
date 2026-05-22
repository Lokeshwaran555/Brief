"""Architecture press scout — ArchDaily, Dezeen, Designboom RSS.

Architects announce projects months before developers do. ArchDaily
and Dezeen routinely publish renderings + practice press releases
60-180 days ahead of the developer's marketing campaign — often the
first public mention of a tower exists in an architecture-press piece.

Strategy: pull each outlet's main RSS, prefilter to entries whose
title or summary mentions a tracked developer or one of the priority
geographies. Drops most international design-blog noise at scout time.

Free, no keys.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx

log = logging.getLogger(__name__)


FEEDS: list[dict[str, str]] = [
    {"name": "ArchDaily",  "url": "https://www.archdaily.com/rss/"},
    {"name": "Dezeen",     "url": "https://www.dezeen.com/feed/"},
    {"name": "Designboom", "url": "https://www.designboom.com/feed/"},
    # World Architecture News — slower cadence but high-end skew.
    {"name": "WAN",        "url": "https://www.worldarchitecturenews.com/feed/"},
]


# Match terms keyed to (region, country_code). First match wins —
# put higher-density terms first.
TARGETS: list[tuple[str, str | None, str | None]] = [
    # Dubai-tracked developers + locations
    ("emaar", "dubai", "AE"),
    ("damac", "dubai", "AE"),
    ("nakheel", "dubai", "AE"),
    ("meraas", "dubai", "AE"),
    ("dubai holding", "dubai", "AE"),
    ("binghatti", "dubai", "AE"),
    ("azizi", "dubai", "AE"),
    ("omniyat", "dubai", "AE"),
    ("ellington", "dubai", "AE"),
    ("sobha", "dubai", "AE"),
    ("dubai", "dubai", "AE"),
    # Abu Dhabi
    ("aldar", "abu_dhabi", "AE"),
    ("modon", "abu_dhabi", "AE"),
    ("mubadala", "abu_dhabi", "AE"),
    ("saadiyat", "abu_dhabi", "AE"),
    ("yas island", "abu_dhabi", "AE"),
    ("reem island", "abu_dhabi", "AE"),
    ("abu dhabi", "abu_dhabi", "AE"),
    # Australia priority
    ("brisbane", "australia", "AU"),
    ("queensland", "australia", "AU"),
    ("mirvac", "australia", "AU"),
    ("stockland", "australia", "AU"),
    ("lendlease", "australia", "AU"),
    # USA priority
    ("austin", "usa", "US"),
    ("dallas", "usa", "US"),
    ("houston", "usa", "US"),
    # Brand-residence partners — match anywhere, region inferred later
    ("aman residences", None, None),
    ("bulgari residences", None, None),
    ("mandarin oriental residences", None, None),
    ("four seasons residences", None, None),
    ("cipriani residences", None, None),
    ("ritz-carlton residences", None, None),
    # Infrastructure / megaproject architecture firms (added 2026-04-26).
    # Major metro + airport + smart-city projects are a leading indicator
    # of where Sobha-relevant land value will shift. These firms publish
    # project portfolios with renderings months before public ground-break.
    ("foster + partners", "dubai", "AE"),
    ("foster and partners", "dubai", "AE"),
    ("aedas", "dubai", "AE"),
    ("aecom", None, None),
    ("killa design", "dubai", "AE"),
    ("zaha hadid", "dubai", "AE"),
    ("som", None, None),  # Skidmore Owings Merrill
]


MAX_AGE_DAYS = 30


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "arch"
    head = (title or "").strip().lower()[:120]
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


def _classify(text: str) -> tuple[str | None, str | None] | None:
    """First-match-wins region tag. Returns None if no match → drop."""
    t = text.lower()
    for needle, region, country in TARGETS:
        if needle in t:
            return (region, country)
    return None


async def _fetch(client: httpx.AsyncClient, feed: dict[str, str]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI architecture scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[arch:%s] fetch failed: %s", feed["name"], e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:40]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        summary = (getattr(e, "summary", "") or "").strip()
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        match = _classify(title + " " + summary)
        if match is None:
            continue
        region, country = match
        out.append(
            {
                "source": f"arch:{feed['name'].lower()}",
                "source_url": link,
                "title": f"{feed['name']} · {title[:240]}",
                "summary": summary[:600],
                "raw_json": {
                    "arch_outlet": feed["name"],
                    "published_date": published,
                    "region": region,
                    "country_code": country,
                    "category_hint": "architecture",
                },
                "dedup_key": _dedup_key(title, link),
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


async def run(limit: int = 25) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch(client, feed) for feed in FEEDS],
            return_exceptions=False,
        )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[architecture] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
