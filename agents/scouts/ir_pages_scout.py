"""Investor Relations pages scout — direct crawl of developer IR feeds.

Most listed real-estate developers post material disclosures on their
IR pages BEFORE the press picks them up. Capital structure changes,
JV announcements, project sell-throughs, leadership moves — all
appear here first when they appear at all.

Strategy by source type:
  - Tier-1 listed devs with public RSS/Atom feeds → direct fetch
  - Devs with structured press-release pages but no RSS → Tavily-mediated
    site search (uses existing Tavily key, free under flat-rate plan)
  - Devs with no public IR page → skip (privately held)

We rely on Tavily for the majority of regional devs because none of
Aldar/Emaar/DAMAC/Mubadala/IHC publish RSS feeds. Tavily's
include_domains filter scopes searches to the IR subdomain so we
don't mix marketing pages with material disclosures.
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

from tools import tavily_search

log = logging.getLogger(__name__)


# ── Tier 1: developers with public IR RSS/Atom feeds ────────────
# Verified live 2026-04-26. ASX-listed AU devs typically have RSS
# from ASX or company site. US REITs often syndicate via PR Newswire.
RSS_FEEDS: list[dict[str, str]] = [
    {
        "name": "Mirvac · ASX announcements",
        # Mirvac (MGR) ASX RSS feed
        "url": "https://www.asx.com.au/asxpdf/rss/MGR.xml",
        "dev_slug": "mirvac",
        "region": "australia",
        "country_code": "AU",
    },
    {
        "name": "Stockland · ASX announcements",
        "url": "https://www.asx.com.au/asxpdf/rss/SGP.xml",
        "dev_slug": "stockland",
        "region": "australia",
        "country_code": "AU",
    },
    {
        "name": "Lendlease · ASX announcements",
        "url": "https://www.asx.com.au/asxpdf/rss/LLC.xml",
        "dev_slug": "lendlease",
        "region": "australia",
        "country_code": "AU",
    },
    {
        "name": "Goodman Group · ASX announcements",
        "url": "https://www.asx.com.au/asxpdf/rss/GMG.xml",
        "dev_slug": "goodman",
        "region": "australia",
        "country_code": "AU",
    },
    # Dubai/AD devs — none have RSS. Handled in TAVILY_DEVS below.
]


# ── Tier 2: developers without RSS — fetch via Tavily site search ──
# Tavily site-restricted search retrieves recent press / news / IR
# announcements scoped to the dev's primary domain. Most reliable way
# to surface IR content for sources that don't expose machine-readable
# feeds.
TAVILY_DEVS: list[dict[str, Any]] = [
    # Dubai (4 tier-1) — corporate sites + DFM disclosure portal.
    # DFM publishes mandatory continuing disclosures for all listed
    # issuers; Tavily can index those even though there's no per-issuer
    # RSS (verified 2026-04-26 — DFM RSS endpoint returns 404).
    {"name": "Emaar",          "domains": ["emaar.com",           "dfm.ae"], "dev_slug": "emaar",     "region": "dubai",     "country_code": "AE"},
    {"name": "DAMAC",          "domains": ["damacproperties.com", "dfm.ae"], "dev_slug": "damac",     "region": "dubai",     "country_code": "AE"},
    {"name": "Nakheel",        "domains": ["nakheel.com"],                   "dev_slug": "nakheel",   "region": "dubai",     "country_code": "AE"},
    {"name": "Dubai Holding",  "domains": ["dubaiholding.com"],              "dev_slug": "dubaih",    "region": "dubai",     "country_code": "AE"},
    {"name": "Union Properties","domains":["unionproperties.com", "dfm.ae"], "dev_slug": "unionprops","region": "dubai",     "country_code": "AE"},
    {"name": "Deyaar",         "domains": ["deyaar.ae",           "dfm.ae"], "dev_slug": "deyaar",    "region": "dubai",     "country_code": "AE"},
    # Abu Dhabi (4 tier-1) — corporate sites + ADX disclosure portal.
    {"name": "Aldar",          "domains": ["aldar.com",           "adx.ae"], "dev_slug": "aldar",     "region": "abu_dhabi", "country_code": "AE"},
    {"name": "Modon",          "domains": ["modon.ae"],                      "dev_slug": "modon",     "region": "abu_dhabi", "country_code": "AE"},
    {"name": "Mubadala",       "domains": ["mubadala.com"],                  "dev_slug": "mubadala",  "region": "abu_dhabi", "country_code": "AE"},
    {"name": "IHC",            "domains": ["ihcgroup.com",        "adx.ae"], "dev_slug": "ihc",       "region": "abu_dhabi", "country_code": "AE"},
    {"name": "RAK Properties", "domains": ["rakproperties.net",   "adx.ae"], "dev_slug": "rakprops",  "region": "abu_dhabi", "country_code": "AE"},
]


# Tavily query intents — same shape as linkedin_scout's QUERY_TEMPLATES.
# Different intent buckets surface different signal types (capital
# moves vs project announcements vs leadership).
QUERY_INTENTS: list[tuple[str, str]] = [
    ("ir-press",
     'press release OR announcement OR "investor update" OR disclosure'),
    ("ir-capital",
     '("capital raise" OR sukuk OR "share buyback" OR dividend OR "joint venture" OR M&A OR acquisition)'),
    ("ir-launch",
     'project launch OR "phase 2" OR "phase 3" OR "master plan" OR "ground breaking" OR handover'),
]

# Discard items older than this from any IR source.
MAX_AGE_DAYS = 30


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "ir"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True  # unknown date → keep, freshness layer handles decay
    try:
        s = str(published_str).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


# ── RSS path ────────────────────────────────────────────────────
async def _fetch_rss(client: httpx.AsyncClient, feed: dict[str, str]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI IR scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[ir:%s] RSS fetch failed: %s", feed["dev_slug"], e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:8]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        out.append(
            {
                "source": f"ir:{feed['dev_slug']}",
                "source_url": link,
                "title": f"{feed['name'].split(' · ')[0]} · {title[:200]}",
                "summary": (getattr(e, "summary", "") or "").strip()[:600],
                "raw_json": {
                    "ir_source": feed["name"],
                    "published_date": published,
                    "region": feed["region"],
                    "country_code": feed["country_code"],
                    "dev_slug_hint": feed["dev_slug"],
                },
                "dedup_key": _dedup_key(title, link),
            }
        )
    return out


# ── Tavily path ─────────────────────────────────────────────────
async def _fetch_tavily(dev: dict[str, Any], intent: tuple[str, str]) -> list[dict[str, Any]]:
    intent_tag, intent_query = intent
    query = f'"{dev["name"]}" {intent_query}'
    try:
        results = await tavily_search.search(
            query,
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=dev["domains"],
            days=30,
        )
    except Exception as e:
        log.warning("[ir:%s/%s] Tavily failed: %s", dev["dev_slug"], intent_tag, e)
        return []

    out: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        published = r.get("published_date")
        if not _within_age(published):
            continue
        out.append(
            {
                "source": f"ir:{dev['dev_slug']}:{intent_tag}",
                "source_url": url,
                "title": f"{dev['name']} IR · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": published,
                    "region": dev["region"],
                    "country_code": dev["country_code"],
                    "dev_slug_hint": dev["dev_slug"],
                    "intent": intent_tag,
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
    return out


# ── Top-level driver ────────────────────────────────────────────
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


async def run(limit: int = 40) -> list[dict[str, Any]]:
    """Fetch RSS feeds + Tavily-mediated IR queries in parallel."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        rss_coros = [_fetch_rss(client, feed) for feed in RSS_FEEDS]
        tavily_coros = [
            _fetch_tavily(dev, intent)
            for dev in TAVILY_DEVS
            for intent in QUERY_INTENTS
        ]
        rss_batches, tavily_batches = await asyncio.gather(
            asyncio.gather(*rss_coros, return_exceptions=False),
            asyncio.gather(*tavily_coros, return_exceptions=False),
        )
    flat = []
    for b in rss_batches:
        flat.extend(b)
    for b in tavily_batches:
        flat.extend(b)
    deduped = _dedupe(flat)
    log.info(
        "[ir_pages] rss=%d tavily=%d total=%d deduped=%d",
        sum(len(b) for b in rss_batches),
        sum(len(b) for b in tavily_batches),
        len(flat), len(deduped),
    )
    return deduped[:limit]
