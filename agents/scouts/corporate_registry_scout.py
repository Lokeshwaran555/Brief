"""Corporate registry scout — Companies House UK + ASIC AU directorships.

Directorship changes at competitor subsidiaries leak strategic moves
weeks before press announcements:
  - New director at "Emaar UK Ltd" → London expansion is real
  - New CEO at "Aldar Australia Pty" → AU push moving forward
  - Director removal across multiple subsidiaries → restructuring

Tavily-mediated since Companies House + ASIC don't expose RSS for
specific entity searches; the data is in their public registers but
indexed by Tavily through reference sites.

Stamp:
  - source: "corporate_registry:<jurisdiction>"
  - category_hint: "competitor"
  - tier_hint: "official"
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


# Each entry: jurisdiction key + Tavily query + domain whitelist.
# UK: Companies House. AU: ASIC. UAE Free Zones: Dubai DIFC, ADGM,
# JAFZA, DMCC don't have public officer-change feeds, so we lean on
# press coverage for those.
# 2026-04-30 stakeholder feedback: directorship records on Companies
# House / ASIC / opencorporates are evergreen — Tavily can't tell
# whether a director was appointed in 2009 or 2026, and the URL leads
# to a filing-history page where ancient appointments mix with new
# ones. We've been surfacing 15-year-old appointments as "news".
#
# Fix: drop the raw-registry domains, keep only PRESS coverage (FT /
# AFR / The National / Estates Gazette / globest etc.) which always
# carries an article date Tavily can filter on. Press also gives the
# context the MD actually wants — "X joined Aldar to lead AU
# expansion" beats "officer appointed at SC719014".
QUERIES: list[dict[str, Any]] = [
    {
        "key":     "uk_press",
        "label":   "UK press · directorship moves",
        "query":   "(appointed OR named OR joins OR hired) director Emaar OR DAMAC OR Aldar OR Sobha OR Nakheel UK real estate 2026",
        "domains": ["ft.com", "thetimes.co.uk", "telegraph.co.uk",
                    "estatesgazette.com", "reuters.com", "bloomberg.com"],
        "country": "GB",
    },
    {
        "key":     "au_press",
        "label":   "Australia press · directorship moves",
        "query":   "(appointed OR named OR joins OR hired) director Mirvac OR Lendlease OR Stockland OR Goodman 2026",
        "domains": ["afr.com", "smh.com.au", "theage.com.au",
                    "theaustralian.com.au", "reuters.com"],
        "country": "AU",
    },
    {
        "key":     "uae_press",
        "label":   "UAE press · subsidiary leadership moves",
        "query":   "(appointed OR named OR joins) Emaar OR DAMAC OR Aldar OR Sobha (CEO OR director OR managing director) 2026 UAE",
        "domains": ["thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "arabianbusiness.com",
                    "zawya.com", "argaam.com", "reuters.com"],
        "country": "AE",
    },
    {
        "key":     "us_press",
        "label":   "US press · RE expansion + leadership",
        "query":   "(appointed OR named OR joins) Emaar OR DAMAC OR Aldar OR Sobha US (CEO OR head OR president) 2026 real estate",
        "domains": ["prnewswire.com", "businesswire.com",
                    "therealdeal.com", "globest.com", "bisnow.com",
                    "wsj.com", "reuters.com"],
        "country": "US",
    },
]


MAX_AGE_DAYS = 90  # quarterly cadence on directorship moves


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "registry"
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
            q["query"], max_results=5,
            topic="general", search_depth="basic",
            include_domains=q["domains"], days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[corporate_registry:%s] tavily failed: %s", q["key"], e)
        return []
    # 2026-04-30 stakeholder feedback: random UK Companies House
    # filings for shell companies with names containing "Aldar" or
    # "Damac" were getting matched to UAE developers in the brief.
    # Filter at the scout: result must mention BOTH a tracked dev
    # name AND a real-estate-anchoring word ("real estate", "property",
    # "developer", "PJSC", "Holdings", "Group"). Otherwise drop.
    DEV_NAMES = ("emaar", "damac", "aldar", "sobha", "nakheel",
                 "binghatti", "azizi", "omniyat", "deyaar", "modon",
                 "mirvac", "lendlease", "stockland", "goodman")
    RE_ANCHOR = ("real estate", "property", "developer", "real-estate",
                 "pjsc", "holdings", "group", "properties",
                 "construction", "homes", "residences",
                 "gulf", "emirates", "uae", "dubai", "abu dhabi")
    # 2026-04-30 fix: require recency words so we don't surface
    # 2009-vintage director appointments. The headline must say it
    # happened — past tense action verbs that imply a fresh event.
    RECENCY_VERBS = (
        "appointed", "named", "joins", "joined", "hired",
        "promoted", "elected", "stepped up", "takes over",
        "to lead", "succeeds", "assumes", "new director",
        "new ceo", "new president", "new managing director",
        "incoming", "to oversee", "to head",
    )
    items: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        snippet = (r.get("content") or "")[:600]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        # Disambiguation: require a dev name AND a real-estate anchor
        # AND a recency verb in title+snippet. Drops UK shell companies
        # AND drops archived officer records ("director since 2009").
        haystack = (title + " " + snippet).lower()
        has_dev = any(d in haystack for d in DEV_NAMES)
        has_anchor = any(a in haystack for a in RE_ANCHOR)
        has_recency = any(v in haystack for v in RECENCY_VERBS)
        if not (has_dev and has_anchor and has_recency):
            continue
        items.append({
            "source":     f"corporate_registry:{q['key']}",
            "source_url": url,
            "title":      f"Registry · {q['label']} · {title[:200]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "jurisdiction":  q["key"],
                "label":         q["label"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "competitor",
                "tier_hint":     "official",
                "region":        {"GB": "other", "AU": "australia",
                                  "AE": "dubai", "US": "usa"}.get(q["country"], "other"),
                "country_code":  q["country"],
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 12) -> list[dict[str, Any]]:
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
    log.info("[corporate_registry] queries=%d fetched=%d deduped=%d",
             len(QUERIES), len(flat), len(out))
    return out[:limit]
