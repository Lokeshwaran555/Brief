"""Construction materials + shipping scout — gross-margin pulse.

Stakeholder ask (2026-04-28): "Construction economics (material prices etc)."

Material prices and container shipping rates are the two biggest movable
inputs to a Sobha project's gross margin. A 12% steel rebar move
or a Red Sea shipping disruption translates directly into AED-per-sqft.

Targets:
  - Steel rebar — Turkey export, China domestic, GCC landed
  - Cement — UAE domestic + global trends
  - Copper, aluminum (LME spot — already in market_snapshot but watching news here)
  - Glass, façade, MEP equipment lead times
  - Diesel — UAE pump price + global Brent spillover
  - Container shipping — SCFI, Drewry WCI, Red Sea status
  - Solar panels + battery pricing (sustainability mandate)

Sources: Trading Economics commodities, S&P Platts, Argus Media,
Drewry, Container News, Hellenic Shipping News, MEED (MENA-specific
construction press), ENR, Construction Dive.

Free under existing Tavily plan. ~6 queries per run.
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


MATERIALS_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "Steel rebar prices",
        "query":   "steel rebar price Turkey export China GCC landed 2026 ton",
        "domains": ["tradingeconomics.com", "spglobal.com", "argusmedia.com",
                    "fastmarkets.com", "metalbulletin.com", "reuters.com",
                    "bloomberg.com", "meed.com", "constructiondive.com"],
        "subcat":  "steel",
    },
    {
        "label":   "Cement + aggregates UAE",
        "query":   "cement price UAE Dubai aggregate construction 2026 ton",
        "domains": ["tradingeconomics.com", "globalcement.com",
                    "cemnet.com", "meed.com", "zawya.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "constructionweekonline.com"],
        "subcat":  "cement",
    },
    {
        "label":   "Copper aluminum LME",
        "query":   "copper aluminum LME spot price 2026 construction electrical",
        "domains": ["tradingeconomics.com", "lme.com", "spglobal.com",
                    "argusmedia.com", "reuters.com", "bloomberg.com",
                    "fastmarkets.com"],
        "subcat":  "base_metals",
    },
    {
        "label":   "Diesel UAE",
        "query":   "diesel price UAE Dubai pump fuel 2026 logistics construction",
        "domains": ["thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "arabianbusiness.com", "zawya.com",
                    "tradingeconomics.com", "reuters.com", "bloomberg.com"],
        "subcat":  "diesel",
    },
    {
        "label":   "Container shipping rates",
        "query":   "SCFI Drewry WCI container shipping rate 2026 Red Sea Suez disruption",
        "domains": ["drewry.co.uk", "container-news.com",
                    "hellenicshippingnews.com", "joc.com",
                    "splash247.com", "reuters.com", "bloomberg.com",
                    "ft.com", "loadstar.co.uk"],
        "subcat":  "shipping",
    },
    {
        "label":   "MEP equipment lead times + solar",
        "query":   "MEP HVAC elevator generator lead time solar panel battery price 2026 construction",
        "domains": ["constructiondive.com", "enr.com", "bdcnetwork.com",
                    "meed.com", "constructionweekonline.com",
                    "pv-magazine.com", "renewableenergyworld.com",
                    "tradingeconomics.com"],
        "subcat":  "mep_solar",
    },
    # 2026-04-29: direct UAE-supplier press / corporate updates.
    {
        "label":   "UAE materials suppliers — direct",
        "query":   "Emirates Steel Conmix Lafarge Holcim UAE 2026 price contract supply",
        "domains": ["emiratesteel.com", "conmix.ae", "lafarge.com",
                    "holcim.com", "constructionweekonline.com",
                    "meed.com", "zawya.com", "thenationalnews.com"],
        "subcat":  "uae_supplier_direct",
    },
]


MAX_AGE_DAYS = 30


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "materials"
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
        log.warning("[materials:%s] Tavily failed: %s", query_def["label"], e)
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
                "source": f"materials:{query_def['subcat']}",
                "source_url": url,
                "title": f"Construction · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": "other",   # global commodities; lives on Construction Econ page
                    "country_code": None,
                    "category_hint": "materials",
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


async def run(limit: int = 20) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(q) for q in MATERIALS_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[materials] queries=%d fetched=%d deduped=%d",
        len(MATERIALS_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
