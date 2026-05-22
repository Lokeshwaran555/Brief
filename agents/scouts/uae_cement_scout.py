"""UAE Cement Manufacturers Association + cement industry tracker.

Domestic cement capacity utilisation and export volumes are leading
indicators of regional construction activity. UAE has 12 major cement
plants; their utilisation drift correlates with project starts 60-90
days ahead.

Tavily-mediated since the Association doesn't publish a clean RSS;
their PDFs land on cement-trade press monthly.

Stamp:
  - source: "uae_cement:<topic>"
  - category_hint: "materials"
  - tier_hint: "press"
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


QUERIES: list[dict[str, Any]] = [
    {
        "label":   "UAE cement capacity + production",
        "query":   "UAE cement production capacity utilisation 2026 monthly Association",
        "domains": ["globalcement.com", "cemnet.com", "cementindustry.co.uk",
                    "constructionweekonline.com", "meed.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "zawya.com"],
        "topic":   "uae_capacity",
    },
    {
        "label":   "UAE cement price",
        "query":   "UAE cement price ton AED 2026 supply demand contractor",
        "domains": ["globalcement.com", "cemnet.com",
                    "constructionweekonline.com", "meed.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "tradingeconomics.com"],
        "topic":   "uae_price",
    },
    {
        "label":   "UAE cement exports",
        "query":   "UAE cement export volume Africa GCC 2026 shipment",
        "domains": ["globalcement.com", "cemnet.com",
                    "constructionweekonline.com", "meed.com",
                    "zawya.com", "thenationalnews.com"],
        "topic":   "uae_exports",
    },
]


MAX_AGE_DAYS = 45


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "uae_cement"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(ts: Any) -> bool:
    if not ts:
        return True
    try:
        s = str(ts).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


async def _fetch_one(q: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            q["query"], max_results=4,
            topic="news", search_depth="basic",
            include_domains=q["domains"], days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[uae_cement:%s] tavily failed: %s", q["topic"], e)
        return []
    items: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        items.append({
            "source":     f"uae_cement:{q['topic']}",
            "source_url": url,
            "title":      f"Cement UAE · {title[:240]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "topic":         q["topic"],
                "label":         q["label"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "materials",
                "tier_hint":     "press",
                "region":        "dubai",
                "country_code":  "AE",
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 8) -> list[dict[str, Any]]:
    batches = await asyncio.gather(*[_fetch_one(q) for q in QUERIES],
                                   return_exceptions=True)
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            continue
        flat.extend(b)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        out.append(it)
    log.info("[uae_cement] queries=%d fetched=%d deduped=%d",
             len(QUERIES), len(flat), len(out))
    return out[:limit]
