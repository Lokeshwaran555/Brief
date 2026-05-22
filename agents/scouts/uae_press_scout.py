"""UAE local-press scout — Khaleej Times, Gulf News, The National, Arabian Business.

Closes the news-freshness gap surfaced by the Gold Line Metro example.
UAE local outlets break Dubai-specific news 6-12h before Google News
indexes it. Until this scout existed the brief was running on lagging
gnews indexing and missing the freshest material moves.

Strategy: pull each outlet's main RSS, prefilter to entries whose
title or summary mentions a tracked developer / sub-market /
infrastructure-or-regulatory term. Drops generic UAE-business noise
(banking earnings, oil prices) at scout time so we don't pay for it
in classifier tokens.

Free RSS, no keys. Defensive code: any RSS URL drift returns [].
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from urllib.parse import quote as _urlquote

log = logging.getLogger(__name__)


def _gn(domain: str, keywords: str, *, gl: str = "AE", hl: str = "en", when: str = "7d") -> str:
    """Google News site-restricted RSS query. 2026-05-18 rewrite:
    publishers' native RSS endpoints are dead (404/403/HTML). Hit
    Google's index instead."""
    q = f'site:{domain} ({keywords}) when:{when}'
    ceid = f"{gl}:{hl}"
    return f"https://news.google.com/rss/search?q={_urlquote(q)}&hl={hl}&gl={gl}&ceid={ceid}"


# Common keyword query used per publisher — Dubai + AD + tracked
# developers + RE moves. Filtered further by KEEP_TERMS below.
_KW = ('Dubai OR "Abu Dhabi" OR UAE) ('
       '"real estate" OR property OR launch OR developer OR '
       'Emaar OR DAMAC OR Aldar OR Nakheel OR Sobha OR Binghatti OR Azizi OR '
       'Modon OR Mubadala OR Meraas OR DLD OR RERA OR "Golden Visa" OR '
       'sukuk OR REIT OR mortgage OR off-plan OR "branded residence"')

FEEDS: list[dict[str, str]] = [
    # Khaleej Times — Dubai's English daily, fastest on RTA / DLD news.
    {"name": "Khaleej Times · UAE business",   "url": _gn("khaleejtimes.com",  _KW), "region": "dubai",     "country_code": "AE"},
    # Gulf News — second daily, often catches Abu Dhabi first.
    {"name": "Gulf News · UAE business",       "url": _gn("gulfnews.com",      _KW), "region": "dubai",     "country_code": "AE"},
    # The National — Abu Dhabi-leaning, breaks AD government news.
    {"name": "The National · AD-leaning",      "url": _gn("thenationalnews.com", _KW), "region": "abu_dhabi", "country_code": "AE"},
    # Arabian Business — pan-MENA, breaks capital + IPO news fast.
    {"name": "Arabian Business · UAE RE",      "url": _gn("arabianbusiness.com", _KW), "region": None,        "country_code": None},
]


# Relevance prefilter — drops generic banking / oil / sports content.
# Mirrors the pattern in policy_wire_scout but with developer + sub-
# market vocabulary so we catch the full RE / infrastructure surface.
KEEP_TERMS: tuple[str, ...] = (
    # Tracked developers
    "emaar", "damac", "aldar", "nakheel", "sobha", "binghatti", "azizi",
    "ellington", "omniyat", "deyaar", "modon", "mubadala", "meraas",
    "dubai holding", "dubai properties", "ihc",
    # Sub-markets (Dubai)
    "hartland", "downtown", "marina", "palm jumeirah", "mbr city",
    "business bay", "creek", "jvc", "jlt", "motor city", "al jaddaf",
    # Sub-markets (Abu Dhabi)
    "saadiyat", "yas island", "reem island", "al maryah", "mariam",
    # Regulatory / capital
    "rera", "dld", "trakheesi", "golden visa", "freehold", "escrow",
    "ejari", "mortgage", "ltv",
    "sukuk", "ipo", "dfm", "adx", "central bank", "real estate",
    # Infrastructure (the Gold Line gap)
    "metro", "gold line", "blue line", "etihad rail", "rta",
    "infrastructure", "megaproject", "smart city", "airport",
    # Generic RE moves
    "launch", "off-plan", "branded residence", "luxury",
    "property", "developer",
)

# Wide age window — catch both breaking same-day news + the 24-72h
# trailing window where a story builds. Older items get filtered by
# the freshness layer in ingest_flow's promotion step.
MAX_AGE_DAYS = 7


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "uae-press"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True
    try:
        s = str(published_str).rstrip("Z")
        # feedparser usually parses to RFC822; we fall back to ISO.
        try:
            dt = datetime.strptime(str(published_str), "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


def _matches_filter(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in KEEP_TERMS)


async def _fetch(client: httpx.AsyncClient, feed: dict[str, str]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI UAE-press scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[uae_press:%s] fetch failed: %s", feed["name"], e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:40]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        summary = re.sub(r"<[^>]+>", " ", (getattr(e, "summary", "") or ""))
        summary = re.sub(r"\s+", " ", summary).strip()
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        if not _matches_filter(title + " " + summary):
            continue
        out.append(
            {
                "source": f"uae_press:{feed['name'].lower().replace(' ', '_')}",
                "source_url": link,
                "title": f"{feed['name'].split(' ')[0]} · {title[:240]}",
                "summary": summary[:600],
                "raw_json": {
                    "outlet": feed["name"],
                    "published_date": published,
                    "region": feed.get("region"),
                    "country_code": feed.get("country_code"),
                    "category_hint": "press",
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


async def run(limit: int = 30) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch(client, f) for f in FEEDS],
            return_exceptions=False,
        )
    flat = [r for batch in batches for r in batch]
    deduped = _dedupe(flat)
    log.info(
        "[uae_press] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
