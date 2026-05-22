"""Australia-specific press scout — free RSS sources only.

Mirrors `ad_press_scout` (2026-05-11) for the Australia region.
Stakeholder feedback: Australia tab routinely empty despite 9 gnews
AU queries + 5 reddit AU subs + 2 GDELT AU queries. Adding a
dedicated specialty press scout to plug the gap with high-yield
RE-focused AU feeds.

All sources are public RSS (no Tavily, no Apify, no API keys).
Each item hard-tags `region=australia, country_code=AU` at scout
time so the dashboard's Australia tab fills regardless of what the
classifier infers from headline alone.

Sources (English, AU-published or AU-focused):
  - News.com.au · Real Estate vertical
  - ABC News · Business + Property tags
  - The Australian · Property
  - AFR (Australian Financial Review) · Property + Companies
  - SMH · Domain RSS
  - Domain.com.au news + research blogs
  - Property Investor magazine RSS
  - Urban Developer (AU industry magazine)
  - Realestate.com.au news

Fail-soft per feed: 404 / network error returns [] for that source.
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
    # News.com.au — broad daily; the RE vertical catches most launches.
    {
        "name": "news.com.au · Real Estate",
        "url":  "https://www.news.com.au/content-feeds/latest-news-real-estate/",
        "dev_slug_hint": None,
    },
    # ABC News Business — picks up AU policy + RE macro coverage.
    {
        "name": "ABC News · Business",
        "url":  "https://www.abc.net.au/news/feed/51892/rss.xml",
        "dev_slug_hint": None,
    },
    # The Australian · Property (paywalled previews, but RSS is free).
    {
        "name": "The Australian · Property",
        "url":  "https://www.theaustralian.com.au/business/property/rss",
        "dev_slug_hint": None,
    },
    # AFR Property + Companies — Tier-1 daily for executive moves.
    {
        "name": "AFR · Property",
        "url":  "https://www.afr.com/rss/property",
        "dev_slug_hint": None,
    },
    {
        "name": "AFR · Companies",
        "url":  "https://www.afr.com/rss/companies",
        "dev_slug_hint": None,
    },
    # Domain — AU's largest RE marketplace + media arm.
    {
        "name": "Domain · News",
        "url":  "https://www.domain.com.au/news/feed/",
        "dev_slug_hint": None,
    },
    # SMH (Sydney Morning Herald) · Property tag.
    {
        "name": "SMH · Business",
        "url":  "https://www.smh.com.au/rss/business.xml",
        "dev_slug_hint": None,
    },
    # The Age · Business (Melbourne paper, Nine network).
    {
        "name": "The Age · Business",
        "url":  "https://www.theage.com.au/rss/business.xml",
        "dev_slug_hint": None,
    },
    # Urban Developer — AU industry magazine, deep developer coverage.
    {
        "name": "Urban Developer",
        "url":  "https://www.theurbandeveloper.com/feed",
        "dev_slug_hint": None,
    },
    # Property Investor Magazine — AU-specific investor pulse.
    {
        "name": "Property Investor",
        "url":  "https://www.propertyinvestor.com.au/feed/",
        "dev_slug_hint": None,
    },
    # Realestate.com.au news/blog (REA Group).
    {
        "name": "REA · News",
        "url":  "https://www.realestate.com.au/news/feed/",
        "dev_slug_hint": None,
    },
    # Brisbane Times Business (Olympic infrastructure coverage).
    {
        "name": "Brisbane Times · Business",
        "url":  "https://www.brisbanetimes.com.au/rss/business.xml",
        "dev_slug_hint": None,
    },
]


# Keep-terms — skewed toward AU anchors + RE moves. The filter
# catches: tracked AU developers, AU sub-markets (Sydney / Melbourne /
# Brisbane / Perth / Gold Coast), Olympic infrastructure, RBA / cash
# rate / mortgage rate / housing affordability, FIRB (foreign
# investment), build-to-rent, plus generic RE moves the classifier
# wants to see.
KEEP_TERMS: tuple[str, ...] = (
    # Tracked AU developers
    "mirvac", "stockland", "lendlease", "goodman group", "charter hall",
    "meriton", "frasers property", "gpt group", "dexus", "crown group",
    "scentre group", "vicinity centres", "mirvac group",
    # AU cities + sub-markets
    "sydney", "melbourne", "brisbane", "perth", "gold coast",
    "adelaide", "canberra", "hobart", "new south wales", "nsw",
    "victoria", "queensland", "qld", "wa", "south australia",
    "olympics 2032", "olympic precinct", "harbour bridge", "barangaroo",
    "docklands", "south bank", "fortitude valley",
    # AU policy + capital
    "rba", "reserve bank of australia", "cash rate", "interest rate",
    "mortgage rate", "housing affordability", "first home buyer",
    "firb", "foreign investment review", "stamp duty", "land tax",
    "asic", "asx", "real estate",
    # Generic RE moves
    "off the plan", "off-the-plan", "apartments", "townhouses",
    "house prices", "auction clearance", "median price",
    "build-to-rent", "btr", "social housing", "affordable housing",
    "launch", "off-plan", "branded residence", "luxury home",
    "property market", "housing market", "developer", "developers",
)


MAX_AGE_DAYS = 7


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "au-press"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True
    try:
        s = str(published_str).rstrip("Z")
        try:
            dt = datetime.strptime(str(published_str), "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


def _relevant(title: str, summary: str) -> bool:
    blob = ((title or "") + " " + (summary or "")).lower()
    return any(t in blob for t in KEEP_TERMS)


async def _fetch_feed(client: httpx.AsyncClient, feed: dict[str, str]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI AU-press scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.info("[au_press:%s] fetch failed: %s", feed["name"], type(e).__name__)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:25]:
        title = (getattr(e, "title", "") or "").strip()
        link  = (getattr(e, "link",  "") or "").strip()
        if not title or not link:
            continue
        summary = (getattr(e, "summary", "") or "").strip()
        if not _relevant(title, summary):
            continue
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        import re as _re
        summary_clean = _re.sub(r"<[^>]+>", "", summary)
        summary_clean = _re.sub(r"\s+", " ", summary_clean).strip()[:600]
        out.append(
            {
                "source":     f"au_press:{feed['name'].split(' · ')[0].lower().replace(' ', '_').replace('.', '')}",
                "source_url": link,
                "title":      title[:300],
                "summary":    summary_clean,
                "raw_json": {
                    "au_press_source": feed["name"],
                    "published_date":  published,
                    # Hard-tag region=australia at scout time so the
                    # dashboard's Australia tab fills without waiting
                    # on the classifier to infer.
                    "region":          "australia",
                    "country_code":    "AU",
                    "dev_slug_hint":   feed.get("dev_slug_hint"),
                },
                "dedup_key":  _dedup_key(title, link),
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


async def run(limit: int = 60) -> list[dict[str, Any]]:
    """Fan-out across AU-leaning RSS feeds in parallel, filter to
    relevant items, dedup across feeds, return up to `limit`.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch_feed(client, f) for f in FEEDS],
            return_exceptions=True,
        )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            log.warning("[au_press] batch raised: %s", b)
            continue
        flat.extend(b)
    deduped = _dedupe(flat)
    log.info(
        "[au_press] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
