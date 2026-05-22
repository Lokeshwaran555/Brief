"""Trade-press scout — Bisnow, Mansion Global, The Real Deal RSS.

Trade press covers commercial + luxury residential with a higher
signal density than mainstream news: deal flow, capital moves,
brokerage stories, and luxury launches that don't make general
news cycles. Mansion Global in particular is the global wire of
record for branded-residences and HNW transactions.

All free RSS, no keys. Defensive code: URL drift returns [].
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
    # Bisnow national + regional. Bisnow's regional editions cover
    # specific MSAs — useful for Texas multifamily.
    {"name": "Bisnow National",      "url": "https://www.bisnow.com/feed",                 "region": "usa",       "country_code": "US"},
    {"name": "Bisnow Texas",         "url": "https://www.bisnow.com/texas/feed",           "region": "usa",       "country_code": "US"},
    {"name": "Bisnow South Florida", "url": "https://www.bisnow.com/south-florida/feed",   "region": "usa",       "country_code": "US"},
    # Mansion Global — luxury / HNW global wire.
    {"name": "Mansion Global",       "url": "https://www.mansionglobal.com/luxury-real-estate-news?id=topic-feed&format=rss", "region": None, "country_code": None},
    # The Real Deal — US commercial + luxury residential trade press.
    {"name": "The Real Deal",        "url": "https://therealdeal.com/feed/",               "region": "usa",       "country_code": "US"},
    # Construction Week (MENA) — UAE construction trade press, often
    # surfaces main-contractor awards before mainstream news.
    {"name": "Construction Week ME", "url": "https://www.constructionweekonline.com/rss",  "region": "dubai",     "country_code": "AE"},
    # MEED — MENA business + projects.
    {"name": "MEED Projects",        "url": "https://www.meed.com/category/projects/feed", "region": "dubai",     "country_code": "AE"},
    # TechCrunch — proptech funding rounds, real-estate startups.
    # Substituted for Crunchbase Daily (Cloudflare-blocked unauth).
    {"name": "TechCrunch",           "url": "https://techcrunch.com/feed/",                "region": None,        "country_code": None},
    # HousingWire — US real estate trade press, mortgage + brokerage.
    {"name": "HousingWire",          "url": "https://www.housingwire.com/feed/",           "region": "usa",       "country_code": "US"},
]


# Match terms keyed to (region, country_code). Used as both relevance
# filter (drops trade-press items unrelated to our watchlist) and
# region tagger when the feed is global (Mansion Global).
TARGETS: list[tuple[str, str | None, str | None]] = [
    # Tracked developers (Dubai)
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
    # Abu Dhabi
    ("aldar", "abu_dhabi", "AE"),
    ("modon", "abu_dhabi", "AE"),
    ("mubadala", "abu_dhabi", "AE"),
    ("ihc ", "abu_dhabi", "AE"),
    # Geographies
    ("dubai", "dubai", "AE"),
    ("abu dhabi", "abu_dhabi", "AE"),
    ("austin", "usa", "US"),
    ("dallas", "usa", "US"),
    ("houston", "usa", "US"),
    ("texas", "usa", "US"),
    ("brisbane", "australia", "AU"),
    ("sydney", "australia", "AU"),
    ("melbourne", "australia", "AU"),
    # US REIT watchlist
    ("camden property", "usa", "US"),
    ("avalonbay", "usa", "US"),
    ("equity residential", "usa", "US"),
    ("essex property", "usa", "US"),
    ("mid-america apartment", "usa", "US"),
    # Branded-residence brands (global)
    ("aman", None, None),
    ("bulgari", None, None),
    ("mandarin oriental", None, None),
    ("four seasons", None, None),
    ("cipriani", None, None),
    ("ritz-carlton", None, None),
    # Luxury / HNW signal terms (Mansion Global default)
    ("branded residence", None, None),
    ("ultra-luxury", None, None),
    ("penthouse", None, None),
    # PropTech / funding-signal terms (TechCrunch default)
    ("proptech", None, None),
    ("real estate startup", None, None),
    ("real estate tech", None, None),
    ("construction tech", None, None),
    ("real-estate fund", None, None),
    ("series a real estate", None, None),
    ("series b real estate", None, None),
]


MAX_AGE_DAYS = 14


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "trade"
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


def _classify(text: str, default_region: str | None, default_country: str | None) -> tuple[str | None, str | None] | None:
    """Match against TARGETS; if nothing matches but the feed has a default
    region (e.g. Bisnow Texas), keep the item with the feed's default tag.
    For region-less feeds (Mansion Global), no match → drop.
    """
    t = text.lower()
    for needle, region, country in TARGETS:
        if needle in t:
            return (region or default_region, country or default_country)
    if default_region:
        return (default_region, default_country)
    return None


async def _fetch(client: httpx.AsyncClient, feed: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI trade-press scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[trade:%s] fetch failed: %s", feed["name"], e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:30]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        summary = (getattr(e, "summary", "") or "").strip()
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        match = _classify(title + " " + summary, feed.get("region"), feed.get("country_code"))
        if match is None:
            continue
        region, country = match
        out.append(
            {
                "source": f"trade:{feed['name'].lower().replace(' ', '_')}",
                "source_url": link,
                "title": f"{feed['name']} · {title[:240]}",
                "summary": summary[:600],
                "raw_json": {
                    "trade_outlet": feed["name"],
                    "published_date": published,
                    "region": region,
                    "country_code": country,
                    "category_hint": "trade_press",
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
            *[_fetch(client, feed) for feed in FEEDS],
            return_exceptions=False,
        )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[trade_press] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
