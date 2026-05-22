"""Sukuk + bond yields scout — UAE sovereign + Sobha + peer developers.

Stakeholder ask (2026-04-28): "Listed comps and capital markets — only
MAG 7, focus on UAE real estate and govt sukuk and bonds."

This is the trickiest data source on the asks list. Live sukuk + CDS
yields are paid feeds (Bloomberg, Refinitiv). We do the next-best
thing: Tavily-mediated weekly research that captures yield moves,
issuance announcements, and rating actions. The dashboard renders an
honest "best-effort" badge on this card so the MD knows the number
came from news, not a market data terminal.

Targets:
  - Sobha Sukuk 2030 (the only listed Sobha instrument)
  - Damac, Emaar, Aldar sukuk yields + spread vs UAE sovereign
  - UAE sovereign sukuk + Abu Dhabi sovereign sukuk
  - Saudi sovereign + PIF issuance (relevant cross-GCC pricing benchmark)
  - Rating actions (Moody's / S&P / Fitch) on Sobha or peers
  - Regional sukuk issuance calendar

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


SUKUK_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "Sobha + UAE developer sukuk yields",
        "query":   "Sobha sukuk 2030 yield Damac Emaar Aldar bond spread 2026",
        "domains": ["bloomberg.com", "reuters.com", "ft.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "zawya.com", "mubasher.info",
                    "arabianbusiness.com", "spglobal.com",
                    "moodys.com", "fitchratings.com"],
        "topic":   "developer_sukuk",
    },
    {
        "label":   "UAE sovereign sukuk",
        "query":   "UAE sovereign sukuk yield Abu Dhabi government bond 2026 issuance",
        "domains": ["bloomberg.com", "reuters.com", "ft.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "zawya.com", "mubasher.info",
                    "spglobal.com", "moodys.com", "fitchratings.com"],
        "topic":   "uae_sovereign",
    },
    {
        "label":   "Saudi PIF sovereign benchmark",
        "query":   "Saudi sovereign sukuk PIF issuance yield 2026 GCC benchmark",
        "domains": ["bloomberg.com", "reuters.com", "ft.com",
                    "arabnews.com", "zawya.com", "mubasher.info",
                    "spglobal.com", "moodys.com", "fitchratings.com"],
        "topic":   "ksa_sovereign",
    },
    {
        "label":   "Rating actions on Sobha + UAE peers",
        "query":   "Moody's S&P Fitch rating action Sobha Damac Emaar Aldar UAE 2026",
        "domains": ["spglobal.com", "moodys.com", "fitchratings.com",
                    "reuters.com", "bloomberg.com", "ft.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "zawya.com", "mubasher.info"],
        "topic":   "rating_action",
    },
    {
        "label":   "Regional sukuk issuance calendar",
        "query":   "GCC MENA sukuk issuance calendar pipeline 2026 listed",
        "domains": ["bloomberg.com", "reuters.com", "ft.com",
                    "zawya.com", "mubasher.info", "thenationalnews.com",
                    "khaleejtimes.com", "arabianbusiness.com"],
        "topic":   "issuance_calendar",
    },
]


MAX_AGE_DAYS = 30  # sukuk moves slower than spot — wider age window


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "sukuk"
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
        log.warning("[sukuk:%s] Tavily failed: %s", query_def["label"], e)
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
                "source": f"sukuk:{query_def['topic']}",
                "source_url": url,
                "title": f"Capital · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": "dubai",          # UAE-focused; sits on Markets & Capital page
                    "country_code": "AE",
                    "category_hint": "capital_markets",
                    "subcategory": "sukuk",
                    "topic": query_def["topic"],
                    "data_quality": "best_effort",  # honest gap: Tavily inference, not Bloomberg
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


async def run(limit: int = 15) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(q) for q in SUKUK_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[sukuk] queries=%d fetched=%d deduped=%d",
        len(SUKUK_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
