"""DFM + ADX exchange disclosures scout (Round-2, 2026-04-29).

ir_pages_scout polls each developer's IR page via Tavily — that catches
announcements when IR teams update their site, often hours-to-days after
the regulatory filing was made. The exchanges themselves publish
mandatory continuing disclosures at the moment of disclosure: material
events, board changes, dividend declarations, share-buyback authorisation,
sukuk issuance, M&A. That's the firehose this scout reads.

DFM does not expose a per-issuer RSS, but their disclosures pages are
public + Tavily-indexable, and they get pinged frequently enough to stay
near-real-time. ADX has a disclosures search at adx.ae.

We hit Tavily with date-bounded queries on the disclosures domain, then
filter to the listed UAE developers we track.

Stamp:
  - source: "dfm_disclosure:<dev_slug>" / "adx_disclosure:<dev_slug>"
  - category_hint: "regulator"  (it IS a regulatory filing)
  - tier_hint: "regulatory"     (highest source tier — official filing)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from tools import tavily_search

log = logging.getLogger(__name__)


# Listed UAE developers we track for material announcements.
LISTED_DEVS: list[dict[str, str]] = [
    # DFM-listed.
    {"name": "Emaar Properties",   "dev_slug": "emaar",       "exchange": "dfm", "region": "dubai"},
    {"name": "DAMAC Properties",   "dev_slug": "damac",       "exchange": "dfm", "region": "dubai"},
    {"name": "Union Properties",   "dev_slug": "unionprops",  "exchange": "dfm", "region": "dubai"},
    {"name": "Deyaar",             "dev_slug": "deyaar",      "exchange": "dfm", "region": "dubai"},
    # ADX-listed.
    {"name": "Aldar Properties",   "dev_slug": "aldar",       "exchange": "adx", "region": "abu_dhabi"},
    {"name": "IHC",                "dev_slug": "ihc",         "exchange": "adx", "region": "abu_dhabi"},
    {"name": "RAK Properties",     "dev_slug": "rakprops",    "exchange": "adx", "region": "abu_dhabi"},
]


# Material-event keywords. Filings about routine ops (changing contact
# info, etc.) get filtered out.
MATERIAL_EVENT_TERMS = (
    "material announcement", "disclosure", "board", "appointment",
    "resignation", "dividend", "interim dividend", "interim financial",
    "annual financial", "quarterly", "earnings", "general assembly", "agm",
    "buyback", "share buyback", "sukuk", "bond", "notes", "issuance",
    "rating", "credit rating", "acquisition", "merger", "joint venture",
    "subsidiary", "capital increase", "share split", "dividend policy",
    "capital reduction",
)


MAX_AGE_DAYS = 7  # Disclosures are time-sensitive; keep window tight.


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "disclosures"
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


def _matches_material(text: str) -> bool:
    t = (text or "").lower()
    return any(term in t for term in MATERIAL_EVENT_TERMS)


async def _fetch_one(dev: dict[str, str]) -> list[dict[str, Any]]:
    domain = "dfm.ae" if dev["exchange"] == "dfm" else "adx.ae"
    query = (
        f'"{dev["name"]}" disclosure announcement '
        f'(material OR dividend OR sukuk OR board OR acquisition OR earnings) '
        f'site:{domain}'
    )
    try:
        results = await tavily_search.search(
            query,
            max_results=5,
            topic="news",
            search_depth="basic",
            include_domains=[domain, "zawya.com", "mubasher.info"],
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[dfm_disclosure:%s] tavily failed: %s", dev["dev_slug"], e)
        return []

    items: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        content = (r.get("content") or "")
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        if not _matches_material(title + " " + content):
            continue
        items.append({
            "source": f"{dev['exchange']}_disclosure:{dev['dev_slug']}",
            "source_url": url,
            "title": f"{dev['exchange'].upper()} · {title[:240]}",
            "summary": content[:600],
            "raw_json": {
                "exchange":    dev["exchange"],
                "dev_slug":    dev["dev_slug"],
                "dev_name":    dev["name"],
                "published_date": r.get("published_date"),
                "tavily_score":   r.get("score"),
                "region":       dev["region"],
                "country_code": "AE",
                "category_hint": "regulator",   # routes to regulatory rail
                "tier_hint":     "regulatory",  # highest source tier
                "dev_slug_hint": dev["dev_slug"],
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 18) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch_one(dev) for dev in LISTED_DEVS],
        return_exceptions=True,
    )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            log.warning("[dfm_disclosures] batch raised: %s", b)
            continue
        flat.extend(b)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        deduped.append(it)
    log.info("[dfm_disclosures] devs=%d fetched=%d deduped=%d",
             len(LISTED_DEVS), len(flat), len(deduped))
    return deduped[:limit]
