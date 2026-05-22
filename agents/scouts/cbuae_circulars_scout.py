"""CBUAE Financial Stability Report + circulars + monthly bulletin.

Existing eibor_scout catches headline EIBOR moves; this scout tracks the
deeper Central Bank publications that signal regulatory regime shifts:
  - Financial Stability Report (semi-annual)
  - Circulars (regulation changes, LTV, mortgage rules, AML)
  - Monthly Statistical Bulletin
  - Notices to banks

Tavily-mediated since CBUAE doesn't publish a public RSS for these.
The signal is RARE but very-high-impact — an LTV cap change moves every
launch's affordability calculus immediately.

Stamp:
  - source: "cbuae_circulars:<topic>"
  - category_hint: "regulator"
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


QUERIES: list[dict[str, Any]] = [
    {
        "label":   "CBUAE LTV / mortgage rules",
        "query":   "UAE Central Bank LTV loan to value mortgage regulation circular 2026",
        "domains": ["centralbank.ae", "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "zawya.com", "mubasher.info",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "topic":   "ltv_mortgage_rules",
    },
    {
        "label":   "CBUAE Financial Stability Report",
        "query":   "UAE Central Bank Financial Stability Report 2026 banking real estate exposure",
        "domains": ["centralbank.ae", "thenationalnews.com", "khaleejtimes.com",
                    "zawya.com", "mubasher.info",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "topic":   "fsr",
    },
    {
        "label":   "CBUAE AML real estate guidance",
        "query":   "UAE Central Bank AML anti money laundering real estate circular guidance 2026",
        "domains": ["centralbank.ae", "thenationalnews.com", "khaleejtimes.com",
                    "zawya.com", "mubasher.info", "reuters.com"],
        "topic":   "aml_real_estate",
    },
    {
        "label":   "CBUAE monetary policy / EIBOR regime",
        "query":   "UAE Central Bank monetary policy EIBOR regime 2026 statement",
        "domains": ["centralbank.ae", "thenationalnews.com", "khaleejtimes.com",
                    "zawya.com", "mubasher.info",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "topic":   "monetary_policy",
    },
]


MAX_AGE_DAYS = 30


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "cbuae"
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
        log.warning("[cbuae_circulars:%s] tavily failed: %s", q["topic"], e)
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
            "source":     f"cbuae_circulars:{q['topic']}",
            "source_url": url,
            "title":      f"CBUAE · {title[:240]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "topic":         q["topic"],
                "label":         q["label"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "regulator",
                "tier_hint":     "regulatory",
                "region":        "dubai",
                "country_code":  "AE",
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 10) -> list[dict[str, Any]]:
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
    log.info("[cbuae_circulars] queries=%d fetched=%d deduped=%d",
             len(QUERIES), len(flat), len(out))
    return out[:limit]
