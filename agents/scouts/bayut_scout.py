"""Bayut listings scout via Apify (epctex/bayut-scraper).

Listings on Bayut are mostly broker-ads — not launches per se — but
they reveal three signals our other sources miss:
  1. New-unit availability spikes by developer/community (broker dump
     = soft launch leak)
  2. Asking-price levels and payment-plan hints in descriptions
  3. Distress signals (price-reduced tags, units that sat for weeks)

We fetch the most-recent listings for each tier-1 developer, emit one
raw signal per listing with the title + price + community + beds
packed into the summary, and let the classifier filter down to
actually interesting ones. The classifier will mostly say keep=false
for run-of-the-mill resales; the survivors are what the MD wants.

Silently returns [] when APIFY_TOKEN is unset.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any
from urllib.parse import urlparse

from tools.apify_client import run_actor

log = logging.getLogger(__name__)

ACTOR = "epctex~bayut-scraper"

# Tier-1 developer-filtered Bayut URLs. These pull the latest units
# advertised under each developer's name. Keep the list short so
# Apify credit doesn't balloon.
START_URLS: list[str] = [
    "https://www.bayut.com/for-sale/property/dubai/?developer=emaar&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=damac&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=sobha&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=nakheel&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=meraas&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=binghatti&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=omniyat&completion_status=off_plan_primary&sort=date_desc",
    "https://www.bayut.com/for-sale/property/dubai/?developer=ellington-properties&completion_status=off_plan_primary&sort=date_desc",
]


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "bayut"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _fmt_summary(item: dict[str, Any]) -> str:
    """Pack the listing's specifics into one line the classifier can score."""
    parts: list[str] = []
    price = item.get("price")
    currency = item.get("currency") or "AED"
    if price:
        parts.append(f"{currency} {price:,}" if isinstance(price, (int, float)) else f"{currency} {price}")
    beds = item.get("rooms") or item.get("bedrooms")
    if beds:
        parts.append(f"{beds}BR")
    area = item.get("area") or item.get("size")
    if area:
        unit = item.get("areaUnit") or "sqft"
        parts.append(f"{area} {unit}")
    loc = item.get("location") or item.get("community") or item.get("address")
    if loc:
        parts.append(str(loc))
    dev = item.get("developer") or item.get("developerName")
    if dev:
        parts.append(f"by {dev}")
    payment = item.get("paymentPlan") or item.get("payment_plan")
    if payment:
        parts.append(f"plan: {payment}")
    return " · ".join(str(p) for p in parts)[:500]


async def run(limit: int = 40) -> list[dict[str, Any]]:
    raw = await run_actor(
        ACTOR,
        {
            "startUrls": [{"url": u} for u in START_URLS],
            "maxItems": 60,
            "proxyConfiguration": {"useApifyProxy": True},
        },
        timeout_seconds=300,
    )
    if not raw:
        return []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in raw:
        title = (it.get("title") or it.get("name") or "").strip()[:300]
        url = (it.get("url") or it.get("link") or "").strip()
        if not title or not url:
            continue
        key = _dedup_key(title, url)
        if key in seen:
            continue
        seen.add(key)
        summary = _fmt_summary(it)
        items.append(
            {
                "source": "bayut:listings",
                "source_url": url,
                "title": title,
                "summary": summary,
                "raw_json": {
                    "price": it.get("price"),
                    "currency": it.get("currency"),
                    "location": it.get("location") or it.get("community"),
                    "developer": it.get("developer") or it.get("developerName"),
                    "beds": it.get("rooms") or it.get("bedrooms"),
                    "area": it.get("area") or it.get("size"),
                    "listing_type": it.get("purpose") or it.get("listingType"),
                    "date": it.get("listedOn") or it.get("createdAt") or it.get("listingDate"),
                    "agent": it.get("agentName") or it.get("agency"),
                },
                "dedup_key": key,
            }
        )
    log.info("[bayut] fetched=%d deduped=%d", len(raw), len(items))
    return items[:limit]
