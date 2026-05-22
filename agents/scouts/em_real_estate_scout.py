"""Emerging-markets real estate scout (Round-3, 2026-04-29).

Tracks the four EM markets where Sobha could plausibly expand or face
new competitor entry:
  - Saudi (REGA / Sakani / WAFI National Housing) — direct competition,
    Vision 2030 capital pipeline
  - Egypt (NUCA / Real Estate Federation) — Sobha already has Egypt
    operations; demand-supply intel relevant
  - Vietnam (Ministry of Construction housing stats / Savills VN) — fast
    growth EM with HNW investor inflows
  - Indonesia (Bank Indonesia residential property survey / Colliers) —
    similar profile

Tavily-mediated. Each agency publishes irregularly; quarterly cadence
is realistic.

Stamp:
  - source: "em_real_estate:<country>"
  - category_hint: "global_prime"
  - tier_hint: "regulatory" for direct agency, "press" for press coverage
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


COUNTRIES: list[dict[str, Any]] = [
    {
        "key":     "saudi",
        "label":   "Saudi Arabia (REGA / Sakani)",
        "query":   "Saudi Arabia REGA Sakani housing residential price index 2026 launches Vision 2030",
        "domains": ["rega.gov.sa", "sakani.sa", "argaam.com",
                    "spa.gov.sa", "tadawul.com.sa",
                    "arabnews.com", "arabianbusiness.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "bloomberg.com", "reuters.com"],
        "country": "SA",
        "region":  "other",
    },
    {
        "key":     "egypt",
        "label":   "Egypt (NUCA / Federation)",
        "query":   "Egypt NUCA new urban communities housing price launches 2026 Cairo",
        "domains": ["nuca.gov.eg", "ahram.org.eg", "egyptindependent.com",
                    "english.aawsat.com", "thenationalnews.com",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "country": "EG",
        "region":  "other",
    },
    {
        "key":     "vietnam",
        "label":   "Vietnam (MoC / Savills)",
        "query":   "Vietnam Ministry Construction housing residential price launches HCMC Hanoi 2026",
        "domains": ["moc.gov.vn", "savills.com.vn", "vir.com.vn",
                    "vietnamnews.vn", "vneconomy.vn",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "country": "VN",
        "region":  "other",
    },
    {
        "key":     "indonesia",
        "label":   "Indonesia (Bank Indonesia / Colliers)",
        "query":   "Indonesia Bank Indonesia residential property survey Jakarta launches 2026",
        "domains": ["bi.go.id", "colliers.com",
                    "thejakartapost.com", "tempo.co",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "country": "ID",
        "region":  "other",
    },
]


MAX_AGE_DAYS = 60


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "em_re"
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


async def _fetch_one(c: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            c["query"], max_results=4,
            topic="news", search_depth="basic",
            include_domains=c["domains"], days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[em_real_estate:%s] tavily failed: %s", c["key"], e)
        return []
    items: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        # Tier inference — direct agency = regulatory, press = press.
        host = (urlparse(url).netloc or "").lower()
        is_agency = any(d in host for d in
                        ("rega.gov.sa", "sakani.sa", "nuca.gov.eg",
                         "moc.gov.vn", "bi.go.id"))
        items.append({
            "source":     f"em_real_estate:{c['key']}",
            "source_url": url,
            "title":      f"EM RE · {c['label']} · {title[:200]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "country":       c["key"],
                "label":         c["label"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "global_prime",
                "tier_hint":     "regulatory" if is_agency else "press",
                "region":        c["region"],
                "country_code":  c["country"],
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 16) -> list[dict[str, Any]]:
    batches = await asyncio.gather(*[_fetch_one(c) for c in COUNTRIES],
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
    log.info("[em_real_estate] countries=%d fetched=%d deduped=%d",
             len(COUNTRIES), len(flat), len(out))
    return out[:limit]
