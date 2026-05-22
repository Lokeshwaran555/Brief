"""Global HPI feeds — city-level price indices (Round-3, 2026-04-29).

Today's `global_prime_scout` returns commentary about prime markets but
no actual time-series. This scout pulls the headline % move + level for
five city HPIs that matter most as substitution-risk for Dubai HNWI
demand. Writes to metrics_daily for charting on Global RE Pulse + emits
a signal when WoW/MoM move is material.

Sources:
  - Rightmove + Nationwide + ONS UK HPI — UK / London prime
  - URA Realis + SRX — Singapore CCR (Core Central Region)
  - StreetEasy data center — NYC Manhattan luxury
  - Centaline + Midland — Hong Kong residential
  - Knight Frank India + ANAROCK — Mumbai/Delhi/Bengaluru luxury

Tavily-mediated. Each index publishes monthly; we fire signal on ≥2%
MoM move (which is meaningful for prime markets).

Stamp:
  - source: "global_hpi:<city_key>"
  - category_hint: "global_prime"
  - tier_hint: "official" (these are statistical agency / portal indices)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from tools import supabase_tool as sb
from tools import tavily_search

log = logging.getLogger(__name__)


CITIES: list[dict[str, Any]] = [
    {
        "key":     "uk_london",
        "label":   "London / UK HPI",
        "query":   "Rightmove Nationwide ONS UK house price index monthly change today",
        "domains": ["rightmove.co.uk", "nationwide.co.uk", "ons.gov.uk",
                    "halifax.co.uk", "lonres.com",
                    "ft.com", "thetimes.co.uk", "telegraph.co.uk",
                    "bloomberg.com", "reuters.com"],
        "country": "GB",
        "fire_threshold_pct": 1.5,
    },
    {
        "key":     "singapore",
        "label":   "Singapore HPI (URA / SRX)",
        "query":   "URA Singapore property price index monthly SRX private residential",
        "domains": ["ura.gov.sg", "srx.com.sg", "edgeprop.sg",
                    "businesstimes.com.sg", "straitstimes.com",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "country": "SG",
        "fire_threshold_pct": 1.5,
    },
    {
        "key":     "nyc_manhattan",
        "label":   "NYC Manhattan luxury HPI",
        "query":   "StreetEasy Manhattan median sale price luxury monthly change Compass Elliman",
        "domains": ["streeteasy.com", "compass.com", "elliman.com",
                    "corcoran.com", "wsj.com", "ft.com",
                    "therealdeal.com", "mansionglobal.com",
                    "bloomberg.com", "reuters.com"],
        "country": "US",
        "fire_threshold_pct": 1.5,
    },
    {
        "key":     "hong_kong",
        "label":   "Hong Kong residential",
        "query":   "Centaline Midland Hong Kong CCL residential property monthly index change",
        "domains": ["centanet.com", "midland.com.hk",
                    "scmp.com", "rvd.gov.hk",
                    "bloomberg.com", "reuters.com", "ft.com"],
        "country": "HK",
        "fire_threshold_pct": 1.5,
    },
    {
        "key":     "india_metro",
        "label":   "India metro luxury",
        "query":   "Knight Frank India ANAROCK Mumbai Bangalore luxury residential price quarterly",
        "domains": ["knightfrank.co.in", "anarock.com",
                    "magicbricks.com", "economictimes.indiatimes.com",
                    "livemint.com", "business-standard.com",
                    "moneycontrol.com"],
        "country": "IN",
        "fire_threshold_pct": 2.0,
    },
]


_PCT_RX = re.compile(
    r"(rose|fell|gained|lost|up|down|jumped|slipped|climbed|dropped|"
    r"increased|decreased|grew|declined)[^\d]{0,30}"
    r"(\d{1,3}(?:\.\d+)?)\s*(%|percent|pct|points?)",
    re.IGNORECASE,
)
_LEVEL_RX = re.compile(
    r"(?:to|at|stands at|reached|index of|level of)\s+"
    r"([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_move(text: str) -> dict[str, Any] | None:
    """Extract a percentage move from press text. 2026-04-30 audit fix:
    we now reject implausibly large MoM moves (>5%) which are almost
    always YoY or annualised figures the LLM mis-tagged as monthly.
    Manhattan HPI 14.4% MoM was the trigger for this guard.
    """
    if not text:
        return None
    pm = _PCT_RX.search(text)
    if not pm:
        return None
    word = pm.group(1).lower()
    try:
        v = float(pm.group(2))
    except Exception:
        return None
    sign = -1 if word in ("fell", "lost", "down", "slipped", "dropped",
                          "decreased", "declined") else +1
    pct = v * sign
    # Sanity guard: HPI MoM moves >5% are almost always YoY mislabels.
    # If the surrounding text doesn't explicitly say YoY/annual, treat
    # the figure as suspect and drop it.
    is_annual = any(t in text.lower() for t in
                    ("yoy", "year-on-year", "year on year",
                     "annual", "annualised", "annualized"))
    if abs(pct) > 5.0 and not is_annual:
        return None
    level: float | None = None
    lm = _LEVEL_RX.search(text)
    if lm:
        try:
            level = float(lm.group(1).replace(",", ""))
        except Exception:
            pass
    return {"pct": pct, "level": level}


async def _fetch_one(city: dict[str, Any]) -> dict[str, Any] | None:
    try:
        results = await tavily_search.search(
            city["query"], max_results=4,
            topic="news", search_depth="basic",
            include_domains=city["domains"],
            days=45,  # monthly indices → wider window
        )
    except Exception as e:
        log.warning("[global_hpi:%s] tavily failed: %s", city["key"], e)
        return None
    # 2026-04-30 audit-round: reject results where the URL or title
    # doesn't actually mention housing/real-estate. Was firing on
    # unrelated press (e.g. "Rallis India earnings" matched on
    # "India HPI" query because Tavily's relevance scoring was
    # surface-level). Title MUST have a housing token.
    HOUSING_TOKENS = (
        "hpi", "house price", "home price", "property price",
        "luxury", "residential", "housing", "real estate",
        "real-estate", "median sale price", "median price",
        "price index", "market index", "manhattan", "rightmove",
        "nationwide", "halifax", "ura", "srx", "centaline",
        "knight frank", "anarock", "elliman", "compass",
        "streeteasy",
    )
    for r in results or []:
        text = (r.get("title") or "") + " " + (r.get("content") or "")
        if not any(tok in text.lower() for tok in HOUSING_TOKENS):
            continue
        s = _parse_move(text)
        if not s:
            continue
        return {
            "key":      city["key"],
            "label":    city["label"],
            "pct":      s.get("pct"),
            "level":    s.get("level"),
            "url":      r.get("url"),
            "title":    (r.get("title") or "")[:300],
            "snippet":  text[:600],
            "published": r.get("published_date"),
        }
    return None


async def run(limit: int = 6) -> list[dict[str, Any]]:
    captured = datetime.now(timezone.utc).replace(hour=0, minute=0,
                                                   second=0, microsecond=0)
    out: list[dict[str, Any]] = []
    for city in CITIES:
        try:
            point = await asyncio.wait_for(_fetch_one(city), timeout=15.0)
        except asyncio.TimeoutError:
            log.info("[global_hpi:%s] per-city timeout", city["key"])
            continue
        if not point:
            continue

        try:
            sb.upsert_metric_daily(
                metric_key=f"hpi_{city['key']}",
                captured_at=captured,
                value_num=point.get("level"),
                source=f"global_hpi:{city['key']}",
                bucket="global",
                unit="index",
                raw_json={
                    "pct_change":  point.get("pct"),
                    "url":         point.get("url"),
                    "title":       point.get("title"),
                    "label":       city["label"],
                    "country":     city["country"],
                },
            )
        except Exception:
            pass

        pct = point.get("pct")
        if pct is None or abs(pct) < city["fire_threshold_pct"]:
            continue

        arrow = "↑" if pct > 0 else "↓"
        title = (
            f"HPI · {city['label']} {arrow}{abs(pct):.1f}% MoM"
            + (f" to {point['level']:,.1f}" if point.get("level") else "")
        )
        summary = (
            f"{city['label']} moved {arrow}{abs(pct):.1f}% in latest reading. "
            f"Substitution-risk read for Dubai HNWI demand — when these "
            f"prime markets accelerate, capital can shift away. "
            f"Source: {point.get('title', '')}."
        )
        dedup_key = hashlib.sha256(
            f"global_hpi:{city['key']}:{captured.date().isoformat()}".encode()
        ).hexdigest()
        out.append({
            "source":     f"global_hpi:{city['key']}",
            "source_url": point.get("url") or "",
            "title":      title[:300],
            "summary":    summary[:600],
            "raw_json": {
                "metric_key":    city["key"],
                "label":         city["label"],
                "pct_change":    pct,
                "level":         point.get("level"),
                "category_hint": "global_prime",
                "tier_hint":     "official",
                "region":        "other",
                "country_code":  city["country"],
            },
            "dedup_key": dedup_key,
        })
        if len(out) >= limit:
            break

    log.info("[global_hpi] cities=%d fired=%d", len(CITIES), len(out))
    return out
