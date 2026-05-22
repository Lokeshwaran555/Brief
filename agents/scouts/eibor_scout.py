"""EIBOR + UAE mortgage rates scout — CBUAE benchmark + bank pricing.

EIBOR (Emirates Interbank Offered Rate) is the benchmark for AED
floating-rate mortgages. A 25bp move shifts the affordability curve
across every Sobha pre-launch in the queue. The CBUAE publishes daily
fixings at https://www.centralbank.ae/en/our-operations/monetary-policy/uae-eibor/
but the page is JS-rendered and Excel-driven — Tavily-mediated query
is the reliable path.

This scout pulls (twice-daily, in line with ingest cadence):
  1. EIBOR 1M / 3M / 6M / 12M most-recent fixings via Tavily
  2. UAE top-bank mortgage rate moves (FAB, ENBD, ADCB, ADIB, RAK Bank)

Each fixing / rate move becomes a signal with category_hint="rate".
The classifier already routes "rate" → macro context, but the
dashboard's Markets & Capital page reads the raw values from the
extended market_snapshot tool too.

Free under the existing Tavily plan. ~5 queries per run.
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


# ── Queries ─────────────────────────────────────────────────────
# EIBOR fixings: pull from CBUAE + Mubasher + The National (which
# republishes CBUAE's daily fixings). Tavily ranks by recency.
EIBOR_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "EIBOR fixings",
        "query":   "EIBOR rate today 1M 3M 6M 12M Central Bank UAE fixing",
        "domains": ["centralbank.ae", "mubasher.info", "thenationalnews.com",
                    "khaleejtimes.com", "gulfnews.com", "zawya.com"],
        "topic":   "EIBOR benchmark fixing",
    },
    {
        "label":   "UAE bank mortgage rates",
        "query":   "UAE mortgage rate FAB ENBD ADCB ADIB RAK Bank home loan rate change",
        "domains": ["bankbazaar.ae", "moneysmart.ae", "yallacompare.com",
                    "khaleejtimes.com", "gulfnews.com", "thenationalnews.com",
                    "zawya.com", "mubasher.info"],
        "topic":   "UAE mortgage pricing",
    },
    {
        "label":   "CBUAE policy / mortgage rules",
        "query":   "UAE Central Bank mortgage cap LTV rule change 2026",
        "domains": ["centralbank.ae", "khaleejtimes.com", "gulfnews.com",
                    "thenationalnews.com", "zawya.com"],
        "topic":   "CBUAE mortgage policy",
    },
]


MAX_AGE_DAYS = 21  # weekly fixings should be fresh; widen a bit for policy items


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "eibor"
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
        log.warning("[eibor:%s] Tavily failed: %s", query_def["label"], e)
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
                "source": f"eibor:{query_def['label'].lower().replace(' ', '_')}",
                "source_url": url,
                "title": f"EIBOR · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": published,
                    "region": "dubai",         # UAE-wide, default to dubai bucket
                    "country_code": "AE",
                    # 2026-04-29: align with themed-page filters. EIBOR + bank
                    # rate moves render on the Markets & Capital page.
                    "category_hint": "capital_markets",
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


async def run(limit: int = 12) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(q) for q in EIBOR_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[eibor] queries=%d fetched=%d deduped=%d",
        len(EIBOR_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
