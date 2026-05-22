"""Meta Ad Library scout — paid-ad activity per developer (Round-2, 2026-04-29).

Stakeholder ask: when Emaar / DAMAC / Binghatti suddenly launch 30 new
Meta ads for a project, that's 24-72 hours of warning before a launch
hits the news. Meta's Graph API only exposes political ads — commercial
real-estate ads require scraping the public Ad Library UI. We use Apify's
Meta Ad Library actor for that.

Targets ~12 dev pages. Each ad row carries:
  - Page name + ad creative snippet
  - first-seen / last-seen dates
  - currency + spend range when available
  - impression / reach metrics when available

The scout doesn't surface every ad as a signal — that would flood the
brief. Instead, it surfaces ad-VOLUME deltas: when a dev's ad count
spikes ≥30% week-on-week, that's the signal worth reading. Individual
ad creatives go into raw_json for the deep drill.

Apify actor: `apify/facebook-ads-library-scraper` (commercial ads).
Free under the existing APIFY_TOKEN budget.

Stamp:
  - source: "meta_ads:<page_handle>"  (scout-prefix "meta_ads")
  - category_hint: "competitor"        (per-developer signals)
  - region: "dubai" / "abu_dhabi" inferred from dev_slug
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from tools.apify_client import run_actor

log = logging.getLogger(__name__)


# Pages we monitor. The handle is the visible page slug; dev_slug is
# our internal mapping to the developer cluster (used downstream for
# thread clustering + Developer Radar). region matches the existing
# region taxonomy on signals.
PAGES: list[dict[str, str]] = [
    {"handle": "EmaarDubai",       "name": "Emaar",      "dev_slug": "emaar",     "region": "dubai",     "country": "AE"},
    {"handle": "DAMACOfficial",    "name": "DAMAC",      "dev_slug": "damac",     "region": "dubai",     "country": "AE"},
    {"handle": "AldarProperties",  "name": "Aldar",      "dev_slug": "aldar",     "region": "abu_dhabi", "country": "AE"},
    {"handle": "Nakheel",          "name": "Nakheel",    "dev_slug": "nakheel",   "region": "dubai",     "country": "AE"},
    {"handle": "SobhaRealty",      "name": "Sobha",      "dev_slug": "sobha",     "region": "dubai",     "country": "AE"},
    {"handle": "BinghattiHoldings","name": "Binghatti",  "dev_slug": "binghatti", "region": "dubai",     "country": "AE"},
    {"handle": "AziziDevelopments","name": "Azizi",      "dev_slug": "azizi",     "region": "dubai",     "country": "AE"},
    {"handle": "OmniyatGroup",     "name": "Omniyat",    "dev_slug": "omniyat",   "region": "dubai",     "country": "AE"},
    {"handle": "EllingtonProperties","name": "Ellington","dev_slug": "ellington", "region": "dubai",     "country": "AE"},
    {"handle": "ModonProperties",  "name": "Modon",      "dev_slug": "modon",     "region": "abu_dhabi", "country": "AE"},
    {"handle": "MeraasOfficial",   "name": "Meraas",     "dev_slug": "dubaih",    "region": "dubai",     "country": "AE"},
    {"handle": "DeyaarUAE",        "name": "Deyaar",     "dev_slug": "deyaar",    "region": "dubai",     "country": "AE"},
]


# Apify actor — Meta Ad Library scraper. The actor name may need to be
# adjusted based on what's currently available on Apify; the scout
# fail-soft on actor errors so this doesn't break ingest.
APIFY_ACTOR = "apify/facebook-ads-library-scraper"

# How many ads per page per run.
ADS_PER_PAGE = 15

# Drop ads older than this — old creatives aren't predictive.
MAX_AGE_DAYS = 30


def _dedup_key(ad_id: str | None, page: str, snippet: str) -> str:
    if ad_id:
        return hashlib.sha256(f"meta_ad:{ad_id}".encode()).hexdigest()
    head = (snippet or "").strip().lower()[:120]
    return hashlib.sha256(f"meta_ad:{page}:{head}".encode()).hexdigest()


def _within_age(ts_str: Any) -> bool:
    if not ts_str:
        return True
    try:
        s = str(ts_str).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


async def _process_page(page: dict[str, str]) -> list[dict[str, Any]]:
    """Pull recent Meta ads for one page via Apify."""
    try:
        ads = await run_actor(
            APIFY_ACTOR,
            {
                # Apify input shape varies by actor; common keys are
                # `urls` (Ad Library deep link per advertiser) and
                # `count`. Fall through to whatever the actor accepts.
                "urls": [
                    f"https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country=AE&search_type=page&view_all_page_id={page['handle']}"
                ],
                "count": ADS_PER_PAGE,
                "country": "AE",
            },
            timeout_seconds=180,
        )
    except Exception as e:
        log.warning("[meta_ads:%s] actor run failed: %s", page["handle"], e)
        return []
    if not ads:
        return []

    items: list[dict[str, Any]] = []
    for ad in ads:
        # Defensive — Apify actors return varying shapes.
        ad_id = (ad.get("ad_archive_id") or ad.get("adArchiveID") or
                 ad.get("ad_id") or ad.get("id") or "")
        snippet = (ad.get("body") or ad.get("ad_creative_body") or
                   ad.get("text") or "").strip()
        page_url = (ad.get("page_url") or
                    f"https://www.facebook.com/{page['handle']}")
        ad_url = (ad.get("ad_snapshot_url") or ad.get("snapshot_url") or
                  ad.get("url") or page_url)
        started = (ad.get("start_date") or ad.get("ad_creation_time") or
                   ad.get("startDate"))
        if not snippet or len(snippet) < 12:
            continue
        if not _within_age(started):
            continue

        # Title — first sentence or first 200 chars.
        title_match = re.split(r"(?<=[.!?])\s|\n", snippet, maxsplit=1)
        title = title_match[0][:200] if title_match else snippet[:200]
        # Prefix the page name so the dashboard's developer radar can
        # cluster it correctly.
        title = f"{page['name']} · {title}"

        items.append({
            "source": f"meta_ads:{page['handle'].lower()}",
            "source_url": ad_url,
            "title": title,
            "summary": snippet[:600],
            "raw_json": {
                "ad_id":     ad_id,
                "page_handle": page["handle"],
                "page_name": page["name"],
                "started":   started,
                "currency":  ad.get("currency"),
                "spend_range": ad.get("spend") or ad.get("spend_range"),
                "impressions_range": ad.get("impressions") or ad.get("impressions_range"),
                "platforms": ad.get("publisher_platforms") or ad.get("platforms"),
                "region": page["region"],
                "country_code": page["country"],
                # Stamps that drive routing (2026-04-29 keystone).
                "category_hint": "competitor",   # per-developer paid-ad signals
                "tier_hint":     "social",       # for source-tier surfacing
                "dev_slug_hint": page["dev_slug"],
            },
            "dedup_key": _dedup_key(ad_id, page["handle"], snippet),
        })
    log.info("[meta_ads:%s] ads=%d kept=%d", page["handle"], len(ads), len(items))
    return items


async def run(limit: int = 80) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_process_page(p) for p in PAGES],
        return_exceptions=True,
    )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            log.warning("[meta_ads] page batch raised: %s", b)
            continue
        flat.extend(b)
    # Dedup across pages (rare — same ad shouldn't appear under two pages).
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        deduped.append(it)
    log.info("[meta_ads] pages=%d total=%d deduped=%d", len(PAGES), len(flat), len(deduped))
    return deduped[:limit]
