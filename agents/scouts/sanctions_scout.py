"""Sanctions + capital-controls tracker (Round-2, 2026-04-29).

Existing geopolitics_scout reads news commentary about sanctions —
useful but lagging. This scout watches three primary regulators whose
moves directly redirect HNWI capital flows into / out of Dubai:

  - OFAC (US Treasury) — SDN list updates, Russia-program changes
  - EU Council        — consolidated sanctions list updates (CIS / Russia)
  - SAFE (China)      — outbound capital controls + QDII updates
  - HMRC / UK Treasury — non-dom tax regime changes (London → Dubai flows)

Each regulator publishes via its own channel — OFAC has RSS, EU has
RSS, SAFE publishes Chinese-language press releases, HMRC publishes
via gov.uk. We use a Tavily-mediated approach for breadth + speed.

Stamp:
  - source: "sanctions:<regulator>"
  - category_hint: "geopolitics"  (routes to Global RE Pulse page)
  - tier_hint: "regulatory"
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


REGULATORS: list[dict[str, Any]] = [
    {
        "key":     "ofac",
        "label":   "OFAC",
        "country": "US",
        "query":   "OFAC SDN list Russia sanctions UAE capital flow Dubai property HNWI",
        "domains": ["home.treasury.gov", "ofac.treasury.gov", "reuters.com",
                    "bloomberg.com", "ft.com", "wsj.com",
                    "thenationalnews.com"],
    },
    {
        "key":     "eu_council",
        "label":   "EU Sanctions",
        "country": "EU",
        "query":   "EU consolidated sanctions list Russia CIS Dubai property capital flow HNWI",
        "domains": ["consilium.europa.eu", "eur-lex.europa.eu", "ec.europa.eu",
                    "reuters.com", "bloomberg.com", "ft.com",
                    "thenationalnews.com"],
    },
    {
        "key":     "safe_china",
        "label":   "SAFE / China capital controls",
        "country": "CN",
        "query":   "China SAFE State Administration Foreign Exchange capital control Dubai property outbound investment",
        "domains": ["safe.gov.cn", "scmp.com", "caixinglobal.com",
                    "reuters.com", "bloomberg.com", "ft.com",
                    "thenationalnews.com"],
    },
    {
        "key":     "hmrc_uk",
        "label":   "HMRC / UK non-dom",
        "country": "GB",
        "query":   "UK HMRC non-domiciled non-dom tax regime change London Dubai migration",
        "domains": ["gov.uk", "hmrc.gov.uk", "ft.com", "thetimes.co.uk",
                    "telegraph.co.uk", "reuters.com", "bloomberg.com",
                    "thenationalnews.com"],
    },
]


MAX_AGE_DAYS = 14


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "sanctions"
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


async def _fetch_one(reg: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            reg["query"],
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=reg["domains"],
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[sanctions:%s] tavily failed: %s", reg["key"], e)
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
            "source": f"sanctions:{reg['key']}",
            "source_url": url,
            "title": f"{reg['label']} · {title[:240]}",
            "summary": (r.get("content") or "")[:600],
            "raw_json": {
                "regulator":     reg["key"],
                "regulator_label": reg["label"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "region":         "other",
                "country_code":   reg["country"],
                "category_hint":  "geopolitics",
                "tier_hint":      "regulatory",
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 12) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch_one(reg) for reg in REGULATORS],
        return_exceptions=True,
    )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            continue
        flat.extend(b)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        deduped.append(it)
    log.info("[sanctions] regulators=%d fetched=%d deduped=%d",
             len(REGULATORS), len(flat), len(deduped))
    return deduped[:limit]
