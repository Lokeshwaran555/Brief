"""Macro + capital flow scout — population growth, FX policy, capital controls.

Stakeholder ask (2026-04-28): "Capital flow, basic indicators like
population growth — that will matter in RE."

Population, migration, and capital-flow indicators are slow-moving but
load-bearing for any real-estate decision over a 3-5 year horizon. This
scout keeps the long-cycle data on the dashboard so the MD doesn't drift
toward only-news thinking.

Targets:
  - UAE population growth (federal NCS, Dubai Statistics Center)
  - Inbound migration to UAE / Dubai (residency stats, Golden Visa flows)
  - Capital flow proxies: India LRS outbound, China outbound investment,
    Russia capital flight, GCC FDI flows
  - Major Indian wedding-season + IPO wealth events (HNWI proxy)
  - World Bank + IMF + UN releases on UAE / GCC / India macro

Sources: World Bank, IMF, UAE NCS, Dubai Statistics Center, RBI,
PBOC, FT, Bloomberg, Reuters, The National, Henley.

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


MACRO_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "UAE population + migration",
        "query":   "UAE population growth 2026 Dubai residents migration Golden Visa stats",
        "domains": ["fcsc.gov.ae", "dsc.gov.ae", "wam.ae",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "arabianbusiness.com", "zawya.com",
                    "worldbank.org", "imf.org", "henleyglobal.com"],
        "subcat":  "uae_population",
        "country_code": "AE",
    },
    {
        "label":   "India LRS outbound capital",
        "query":   "India LRS Liberalised Remittance Scheme overseas remittance 2026 RBI cap",
        "domains": ["rbi.org.in", "economictimes.indiatimes.com",
                    "livemint.com", "business-standard.com",
                    "moneycontrol.com", "ft.com", "bloomberg.com",
                    "reuters.com", "thenationalnews.com"],
        "subcat":  "india_lrs",
        "country_code": "IN",
    },
    {
        "label":   "China outbound investment",
        "query":   "China outbound direct investment ODI 2026 capital outflow real estate",
        "domains": ["scmp.com", "ft.com", "bloomberg.com",
                    "reuters.com", "wsj.com", "caixinglobal.com",
                    "thenationalnews.com"],
        "subcat":  "china_odi",
        "country_code": "CN",
    },
    {
        "label":   "GCC FDI + sovereign wealth",
        "query":   "GCC UAE foreign direct investment FDI sovereign wealth fund 2026 inflow outflow",
        "domains": ["worldbank.org", "imf.org", "oecd.org",
                    "ft.com", "bloomberg.com", "reuters.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "arabianbusiness.com", "zawya.com"],
        "subcat":  "gcc_fdi",
        "country_code": "AE",
    },
    {
        "label":   "Inflation + central bank backdrop",
        "query":   "UAE India US inflation CPI 2026 central bank rate decision real estate",
        "domains": ["centralbank.ae", "rbi.org.in", "federalreserve.gov",
                    "ft.com", "bloomberg.com", "reuters.com",
                    "wsj.com", "imf.org", "worldbank.org",
                    "thenationalnews.com"],
        "subcat":  "inflation",
        "country_code": None,
    },
]


MAX_AGE_DAYS = 45


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "macro"
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
        log.warning("[macro:%s] Tavily failed: %s", query_def["label"], e)
        return []

    out: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        # UAE-tagged macro lands in the dubai bucket so it surfaces under
        # MD Scan's Dubai region tab; everything else is global.
        region = "dubai" if query_def.get("country_code") == "AE" else "other"
        out.append(
            {
                "source": f"macro:{query_def['subcat']}",
                "source_url": url,
                "title": f"Macro · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": region,
                    "country_code": query_def.get("country_code"),
                    "category_hint": "capital_flow",
                    "subcategory": query_def["subcat"],
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
        *[_fetch(q) for q in MACRO_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[macro] queries=%d fetched=%d deduped=%d",
        len(MACRO_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
