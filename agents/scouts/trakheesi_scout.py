"""Trakheesi / DLD project-registry scout.

Trakheesi (trakheesi.dubailand.gov.ae) is the RERA off-plan project
registry. Every off-plan project must register here — with start
dates, completion targets, escrow numbers, unit counts — *before*
marketing begins. Entries appear weeks before any press release.

Direct HTML scraping of Trakheesi is on the roadmap but needs
interactive iteration on the search-results pagination. For now we
go Tavily-mediated, scoped to `dubailand.gov.ae` and adjacent DLD
properties, keyed on registration / escrow / project-launch
vocabulary. This catches:
  - Trakheesi project listings indexed by Tavily
  - DLD news pages announcing new project registrations
  - RERA notices about escrow account openings
  - Property Monitor / DXBInteract press releases that cite
    fresh DLD project IDs

Free under the existing Tavily flat-rate plan.
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


# Domains to search. dubailand.gov.ae covers DLD + Trakheesi +
# RERA pages. Added the property-data aggregators that ingest DLD
# data + reflect it back faster than the source itself.
DOMAINS: list[str] = [
    "dubailand.gov.ae",
    "trakheesi.dubailand.gov.ae",
    "dxbinteract.com",
    "propertymonitor.com",
    "reidin.com",
]


# Intent buckets — each surfaces a different stage of the project
# lifecycle. Run separately so Tavily ranks each intent independently.
QUERY_INTENTS: list[tuple[str, str]] = [
    ("registration",
     '("project registered" OR "Trakheesi" OR "DLD project ID" OR "RERA approval" OR "permit issued") Dubai'),
    ("escrow",
     '("escrow account" OR "escrow opened" OR "oqood issued" OR "trust account") Dubai project'),
    ("launch-window",
     '("off-plan launch" OR "new project announced" OR "marketing approval" OR "OQOOD") Dubai 2026'),
    ("completion",
     '("project completion" OR "Building Completion Certificate" OR "BCC issued" OR "fire safety approval" OR "DEWA energization") Dubai'),
    ("status-change",
     '("project status" OR "phase update" OR "phase 2 registered" OR "phase 3 registered" OR "extended completion") Dubai'),
]


MAX_AGE_DAYS = 30


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "trakheesi"
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


async def _fetch(intent: tuple[str, str]) -> list[dict[str, Any]]:
    intent_tag, query = intent
    try:
        results = await tavily_search.search(
            query,
            max_results=5,
            topic="news",
            search_depth="basic",
            include_domains=DOMAINS,
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[trakheesi:%s] Tavily failed: %s", intent_tag, e)
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
                "source": f"trakheesi:{intent_tag}",
                "source_url": url,
                "title": f"DLD/Trakheesi · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": published,
                    "region": "dubai",
                    "country_code": "AE",
                    "intent": intent_tag,
                    "category_hint": "registry",
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


async def run(limit: int = 25) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(intent) for intent in QUERY_INTENTS],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[trakheesi] intents=%d fetched=%d deduped=%d",
        len(QUERY_INTENTS), len(flat), len(deduped),
    )
    return deduped[:limit]
