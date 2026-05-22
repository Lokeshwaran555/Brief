"""Industry research scout — quarterly market reports from JLL, CBRE,
Knight Frank, Savills, Cushman & Wakefield, Bayut, CoreLogic.

These firms publish dense, structured research with PSF, absorption,
supply-pipeline, and yield data. The reports are gold for fact
extraction — when an investigation references 'JLL says Dubai prime
PSF rose 14% YoY', the user can trust that more than a tweet.

Strategy: Tavily site-restricted search per region per firm. We
let Tavily handle the actual fetching + content extraction since
the report PDFs are usually behind landing pages with tracking
forms. Tavily's `topic=news` and `days` filters keep the recency
bar high.

All free under the existing Tavily flat-rate plan. Twice-daily run
hits ~7 regions × ~3 firms = ~21 queries — well under our budget.
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


# ── Research firms by region ────────────────────────────────────
# Each entry: firm_name, domain(s), regions covered
FIRMS: list[dict[str, Any]] = [
    {
        "name": "JLL",
        "domains": ["jll.com", "jll.ae", "jll.com.au"],
        "regions": ["dubai", "abu_dhabi", "usa", "australia", "other"],
    },
    {
        "name": "CBRE",
        "domains": ["cbre.com", "cbre.ae", "cbre.com.au"],
        "regions": ["dubai", "abu_dhabi", "usa", "australia", "other"],
    },
    {
        "name": "Knight Frank",
        "domains": ["knightfrank.com", "knightfrank.ae", "knightfrank.com.au"],
        "regions": ["dubai", "abu_dhabi", "usa", "australia", "other"],
    },
    {
        "name": "Savills",
        "domains": ["savills.com", "savills.ae"],
        "regions": ["dubai", "abu_dhabi", "australia", "other"],
    },
    {
        "name": "Cushman & Wakefield",
        "domains": ["cushmanwakefield.com"],
        "regions": ["dubai", "abu_dhabi", "usa", "australia", "other"],
    },
    {
        "name": "Bayut",
        "domains": ["bayut.com"],
        "regions": ["dubai", "abu_dhabi"],
    },
    {
        "name": "Property Finder",
        "domains": ["propertyfinder.ae"],
        "regions": ["dubai", "abu_dhabi"],
    },
    {
        "name": "CoreLogic",
        "domains": ["corelogic.com.au", "corelogic.com"],
        "regions": ["australia", "usa"],
    },
    {
        "name": "NMHC",  # National Multifamily Housing Council
        "domains": ["nmhc.org"],
        "regions": ["usa"],
    },
]


# Region → query template. The MD cares about market-state numbers
# (PSF, absorption, supply pipeline, yields) and anything Q2 2026.
REGION_QUERIES: dict[str, str] = {
    "dubai":     "Dubai real estate market report 2026 prime PSF absorption supply",
    "abu_dhabi": "Abu Dhabi real estate market report 2026 PSF absorption Saadiyat Reem",
    "usa":       "US multifamily real estate market report 2026 occupancy rent growth absorption",
    "australia": "Australia residential real estate report 2026 Brisbane Sydney Melbourne supply",
    "other":     "global luxury real estate market report 2026 prime international",
}


MAX_AGE_DAYS = 120  # quarterlies are slow-publishing — broader than scout norm


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "research"
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


async def _fetch(firm: dict[str, Any], region: str) -> list[dict[str, Any]]:
    if region not in firm["regions"]:
        return []
    query = f"\"{firm['name']}\" {REGION_QUERIES.get(region, REGION_QUERIES['other'])}"
    try:
        results = await tavily_search.search(
            query,
            max_results=3,
            topic="news",
            search_depth="basic",
            include_domains=firm["domains"],
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[research:%s/%s] Tavily failed: %s", firm["name"], region, e)
        return []

    region_country = {
        "dubai": "AE", "abu_dhabi": "AE", "usa": "US",
        "australia": "AU", "other": None,
    }
    out: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        published = r.get("published_date")
        if not _within_age(published):
            continue
        # Filter out generic landing pages — research reports usually
        # have year + region in their URL or title. Loose heuristic.
        if not any(tok in (title.lower() + url.lower()) for tok in ("2026", "2025", "report", "outlook", "review", "trends", "research", "insights")):
            continue
        out.append(
            {
                "source": f"research:{firm['name'].lower().replace(' ', '_')}",
                "source_url": url,
                "title": f"{firm['name']} · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": published,
                    "region": region,
                    "country_code": region_country.get(region),
                    "research_firm": firm["name"],
                    "category_hint": "research",
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


async def run(limit: int = 30) -> list[dict[str, Any]]:
    """Fan out across (firm × region) pairs in parallel via Tavily."""
    pairs = [(firm, region) for firm in FIRMS for region in firm["regions"]]
    batches = await asyncio.gather(
        *[_fetch(firm, region) for firm, region in pairs],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[industry_research] pairs=%d fetched=%d deduped=%d",
        len(pairs), len(flat), len(deduped),
    )
    return deduped[:limit]
