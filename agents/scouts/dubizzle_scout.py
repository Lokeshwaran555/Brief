"""Dubizzle listings scout via Apify.

Dubizzle skews slightly more mainstream/end-user vs Bayut's broker-heavy
mix. Running both captures listings that appear on one and not the other
(and lets the classifier see volume divergences between platforms as a
separate signal).

Silently returns [] when APIFY_TOKEN is unset.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any
from urllib.parse import urlparse

from tools.apify_client import run_actor

log = logging.getLogger(__name__)

ACTOR = "epctex~dubizzle-scraper"

START_URLS: list[str] = [
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=emaar",
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=damac",
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=sobha",
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=nakheel",
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=meraas",
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=binghatti",
    "https://dubai.dubizzle.com/en/property-for-sale/residential/?is_off_plan=1&keywords=aldar",
]


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "dubizzle"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _fmt_summary(item: dict[str, Any]) -> str:
    parts: list[str] = []
    price = item.get("price")
    if price:
        parts.append(f"AED {price:,}" if isinstance(price, (int, float)) else f"AED {price}")
    beds = item.get("bedrooms") or item.get("rooms")
    if beds:
        parts.append(f"{beds}BR")
    area = item.get("area") or item.get("size")
    if area:
        parts.append(f"{area} sqft")
    loc = item.get("location") or item.get("neighborhood") or item.get("address")
    if loc:
        parts.append(str(loc))
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
        items.append(
            {
                "source": "dubizzle:listings",
                "source_url": url,
                "title": title,
                "summary": _fmt_summary(it),
                "raw_json": {
                    "price": it.get("price"),
                    "location": it.get("location") or it.get("neighborhood"),
                    "beds": it.get("bedrooms") or it.get("rooms"),
                    "area": it.get("area") or it.get("size"),
                    "date": it.get("createdAt") or it.get("postedDate") or it.get("listingDate"),
                    "agent": it.get("agentName") or it.get("seller"),
                },
                "dedup_key": key,
            }
        )
    log.info("[dubizzle] fetched=%d deduped=%d", len(raw), len(items))
    return items[:limit]
