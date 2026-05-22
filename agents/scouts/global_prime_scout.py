"""Global RE prime markets scout — substitution risk for Dubai HNWI demand.

Stakeholder ask (2026-04-28): "Global real estate news, growth, etc."

Dubai's HNW buyer pool is global; if London prime, Singapore prime,
or NYC luxury heat up, capital substitutes away from Dubai. This scout
captures the prime-market pulse for the cities that matter most as
substitutes.

Targets:
  - London prime (Knight Frank PCL index, Mayfair, Knightsbridge)
  - Singapore prime (CCR, ABSD impact)
  - NYC + Miami prime (Manhattan, Hamptons, Miami Beach, Palm Beach)
  - Monaco + Côte d'Azur
  - Mumbai + Bangalore luxury (key Indian comparator for INR-AED flow)
  - Hong Kong luxury (substitution from China outbound)
  - Tokyo prime (newer competitor)
  - HNWI migration reports (Henley, Knight Frank Wealth Report)

Sources: Knight Frank, Savills, Christie's International, JLL,
Mansion Global, FT, WSJ, Bloomberg, Property Week, Henley.

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


PRIME_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "London prime",
        "query":   "London prime central residential property index Mayfair Knightsbridge 2026 PCL",
        "domains": ["knightfrank.com", "savills.com", "ft.com",
                    "bloomberg.com", "wsj.com", "mansionglobal.com",
                    "propertyweek.com", "thetimes.co.uk", "telegraph.co.uk",
                    "estatesgazette.com"],
        "city":    "london",
        "country_code": "GB",
    },
    {
        "label":   "Singapore prime",
        "query":   "Singapore prime CCR luxury condo property 2026 ABSD",
        "domains": ["knightfrank.com", "savills.com", "ft.com",
                    "bloomberg.com", "businesstimes.com.sg",
                    "straitstimes.com", "mansionglobal.com",
                    "propertyguru.com.sg", "edgeprop.sg"],
        "city":    "singapore",
        "country_code": "SG",
    },
    {
        "label":   "NYC + Miami prime",
        "query":   "Manhattan luxury condo NYC Hamptons Miami Beach Palm Beach prime 2026",
        "domains": ["knightfrank.com", "christiesrealestate.com",
                    "wsj.com", "ft.com", "bloomberg.com",
                    "mansionglobal.com", "therealdeal.com",
                    "miami.curbed.com", "nytimes.com"],
        "city":    "nyc_miami",
        "country_code": "US",
    },
    {
        "label":   "Monaco + Riviera",
        "query":   "Monaco Cote d'Azur Riviera luxury property prime 2026 transaction",
        "domains": ["knightfrank.com", "savills.com", "christiesrealestate.com",
                    "ft.com", "bloomberg.com", "mansionglobal.com",
                    "monacopropertieslist.com", "monacolife.net"],
        "city":    "monaco",
        "country_code": "MC",
    },
    {
        "label":   "Mumbai + Bangalore luxury",
        "query":   "Mumbai Bangalore luxury residential property 2026 PSF absorption",
        "domains": ["knightfrank.co.in", "anarock.com", "jll.co.in",
                    "cbre.co.in", "economictimes.indiatimes.com",
                    "livemint.com", "mansionglobal.com", "ft.com"],
        "city":    "india_metro",
        "country_code": "IN",
    },
    {
        "label":   "HNWI migration + wealth flows",
        "query":   "HNWI millionaire migration 2026 Dubai London Singapore Henley Knight Frank wealth report",
        "domains": ["henleyglobal.com", "knightfrank.com",
                    "ft.com", "bloomberg.com", "wsj.com",
                    "thenationalnews.com", "mansionglobal.com",
                    "forbes.com", "businessinsider.com"],
        "city":    "global",
        "country_code": None,
    },
    # 2026-04-29: direct prime-broker research + insights pages.
    {
        "label":   "Prime brokers — direct research",
        "query":   "prime residential market report 2026 PSF transaction Knight Frank Savills Christie's research insights",
        "domains": ["knightfrank.com", "savills.com",
                    "christiesrealestate.com", "sothebysrealty.com",
                    "jll.com", "cbre.com"],
        "city":    "broker_direct",
        "country_code": None,
    },
]


MAX_AGE_DAYS = 45  # prime indices publish quarterly — slow-moving, wider window


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "global_prime"
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
        log.warning("[global_prime:%s] Tavily failed: %s", query_def["label"], e)
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
                "source": f"global_prime:{query_def['city']}",
                "source_url": url,
                "title": f"Global · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": "other",  # all non-AE/US/AU prime → Global RE Pulse page
                    "country_code": query_def.get("country_code"),
                    "category_hint": "global_prime",
                    "city_hint": query_def["city"],
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
        *[_fetch(q) for q in PRIME_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[global_prime] queries=%d fetched=%d deduped=%d",
        len(PRIME_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
