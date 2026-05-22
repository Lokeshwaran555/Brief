"""Sydney + Melbourne auction clearance scout — weekly market depth.

2026-04-28 stakeholder feedback: "Australia is not coming properly,
Sydney also to be added." The existing au_building_scout pulls only
ABS state-level aggregates (Tables 01/03 dwelling approvals); that
gives the headline but not auction-day data.

This scout adds the *demand-side* signal AU brokers care about:
  - REINSW + REIV weekly auction clearance rates (Sydney + Melbourne)
  - Domain.com.au + realestate.com.au weekly auction wraps
  - Suburb-level auction results for Sobha-relevant submarkets:
    Brunswick (Melbourne), Drummoyne (Sydney), and the broader
    inner Sydney + inner Melbourne rings

Strategy: Tavily site-restricted queries. The portals publish
weekly summary articles ("Sydney clearance rate 68% this week,
Drummoyne sold above reserve"). We capture those, not the
per-listing CRUD.

Free under existing Tavily plan. ~5 queries per run.
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


AU_QUERIES: list[dict[str, Any]] = [
    {
        "label":     "Sydney auction clearance",
        "query":     "Sydney auction clearance rate weekly result median 2026",
        "domains":   ["domain.com.au", "realestate.com.au", "reinsw.com.au",
                      "afr.com", "abc.net.au", "smh.com.au",
                      "corelogic.com.au"],
        "city_hint": "sydney",
    },
    {
        "label":     "Melbourne auction clearance",
        "query":     "Melbourne auction clearance rate weekly result median 2026",
        "domains":   ["domain.com.au", "realestate.com.au", "reiv.com.au",
                      "afr.com", "abc.net.au", "theage.com.au",
                      "corelogic.com.au"],
        "city_hint": "melbourne",
    },
    {
        "label":     "Drummoyne Sydney inner-west",
        "query":     "Drummoyne Sydney inner west property auction price 2026",
        "domains":   ["domain.com.au", "realestate.com.au",
                      "afr.com", "smh.com.au"],
        "city_hint": "sydney_drummoyne",
    },
    {
        "label":     "Brunswick Melbourne inner-north",
        "query":     "Brunswick Melbourne inner north property auction price 2026",
        "domains":   ["domain.com.au", "realestate.com.au",
                      "afr.com", "theage.com.au"],
        "city_hint": "melbourne_brunswick",
    },
    {
        "label":     "RBA rate decision AU mortgage",
        "query":     "RBA cash rate decision Australian mortgage 2026 outlook",
        "domains":   ["rba.gov.au", "afr.com", "abc.net.au", "smh.com.au",
                      "theage.com.au", "reuters.com", "bloomberg.com"],
        "city_hint": "national",
    },
]


MAX_AGE_DAYS = 14


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "auctions_au"
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
        log.warning("[sydney_auctions:%s] Tavily failed: %s", query_def["label"], e)
        return []

    out: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        out.append(
            {
                "source": f"au_auctions:{query_def['city_hint']}",
                "source_url": url,
                "title": f"AU · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": "australia",
                    "country_code": "AU",
                    "category_hint": "demand",  # auction clearance = demand signal
                    "city_hint": query_def["city_hint"],
                    "topic": query_def["label"],
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


async def run(limit: int = 18) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(q) for q in AU_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[sydney_auctions] queries=%d fetched=%d deduped=%d",
        len(AU_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
