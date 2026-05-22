"""LME / SHFE / CME daily metals settlement scout (Round-3, 2026-04-29).

The single most important upstream input for tower construction is steel
rebar — copper and aluminum are next. Today's `materials_scout` infers
prices via Tavily news commentary; this scout pulls actual settlement
levels and writes them to metrics_daily so the Construction Econ page
can chart them as time-series.

Sources:
  - LME (London Metal Exchange) — copper, aluminum daily ring close.
    Public delayed data via lme.com/en/Metals.
  - SHFE (Shanghai Futures Exchange) — steel rebar (RB), hot-rolled coil
    (HC), copper (CU), aluminum (AL). Daily settlement CSV at shfe.com.cn.
  - CME — HRC steel futures, iron ore (62% Fe). Free settlement.

Implementation: Tavily-mediated extraction (same as cme_fedwatch_scout)
because the direct CSVs need auth or shift formats. Specialized news
sources publish the closes daily in headlines we can parse.

Stamp:
  - source: "lme_metals:<metric_key>"
  - category_hint: "materials" (routes to Construction Econ page)
  - tier_hint: "press"
  - metrics_daily: metric_key="metals_<symbol>", bucket="construction"
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


METALS: list[dict[str, Any]] = [
    {
        "key":        "steel_rebar",
        "label":      "Steel rebar (SHFE RB)",
        "query":      "SHFE rebar steel daily settlement price RMB China today",
        "domains":    ["shfe.com.cn", "fastmarkets.com", "metalbulletin.com",
                       "spglobal.com", "argusmedia.com",
                       "reuters.com", "bloomberg.com", "ft.com",
                       "splash247.com"],
        "unit":       "RMB/t",
        "fire_threshold_pct": 3.0,  # ≥3% daily move → fire signal
    },
    {
        "key":        "copper",
        "label":      "Copper (LME)",
        "query":      "LME copper daily settlement price USD per ton today close",
        "domains":    ["lme.com", "fastmarkets.com", "spglobal.com",
                       "argusmedia.com", "reuters.com", "bloomberg.com",
                       "ft.com", "tradingeconomics.com"],
        "unit":       "USD/t",
        "fire_threshold_pct": 3.0,
    },
    {
        "key":        "aluminum",
        "label":      "Aluminum (LME)",
        "query":      "LME aluminum daily settlement price USD per ton today close",
        "domains":    ["lme.com", "fastmarkets.com", "spglobal.com",
                       "argusmedia.com", "reuters.com", "bloomberg.com",
                       "ft.com", "tradingeconomics.com"],
        "unit":       "USD/t",
        "fire_threshold_pct": 3.0,
    },
    {
        "key":        "iron_ore",
        "label":      "Iron Ore 62% Fe (SGX)",
        "query":      "iron ore 62% Fe SGX daily settlement price USD ton today",
        "domains":    ["sgx.com", "fastmarkets.com", "spglobal.com",
                       "argusmedia.com", "reuters.com", "bloomberg.com"],
        "unit":       "USD/t",
        "fire_threshold_pct": 4.0,  # iron ore is more volatile
    },
    {
        "key":        "hrc_steel",
        "label":      "HRC steel (CME)",
        "query":      "CME HRC steel hot rolled coil daily settlement price USD ton",
        "domains":    ["cmegroup.com", "fastmarkets.com", "spglobal.com",
                       "argusmedia.com", "reuters.com", "bloomberg.com"],
        "unit":       "USD/t",
        "fire_threshold_pct": 3.0,
    },
]


# Match patterns like:
#   "Copper closed at $9,415/t (+1.2%)"
#   "Steel rebar settled at 3,720 yuan, down 2.1%"
#   "Iron ore rose 1.5% to $112.30 per tonne"
_LEVEL_RX = re.compile(
    r"(?:closed?|settled?|stood at|reached|trading at|rose to|fell to|at)\s+"
    r"(?:[A-Z]{0,4}\$|RMB|¥|USD|yuan)?\s*"
    r"([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_PCT_RX = re.compile(
    r"(rose|fell|gained|lost|up|down|jumped|slipped|climbed|dropped|"
    r"increased|decreased|added|shed)[^\d]{0,30}"
    r"(\d{1,3}(?:\.\d+)?)\s*(%|percent|pct)",
    re.IGNORECASE,
)


def _parse_settlement(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    level_match = _LEVEL_RX.search(text)
    pct_match = _PCT_RX.search(text)
    level: float | None = None
    if level_match:
        try:
            level = float(level_match.group(1).replace(",", ""))
        except Exception:
            level = None
    pct: float | None = None
    if pct_match:
        word = pct_match.group(1).lower()
        try:
            v = float(pct_match.group(2))
        except Exception:
            v = None
        if v is not None:
            sign = -1 if word in ("fell", "lost", "down", "slipped",
                                  "dropped", "decreased", "shed") else +1
            pct = v * sign
    if level is None and pct is None:
        return None
    return {"level": level, "pct": pct}


async def _fetch_one(metal: dict[str, Any]) -> dict[str, Any] | None:
    try:
        results = await tavily_search.search(
            metal["query"],
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=metal["domains"],
            days=4,  # tight window — we want today/yesterday's close
        )
    except Exception as e:
        log.warning("[lme_metals:%s] tavily failed: %s", metal["key"], e)
        return None
    for r in results or []:
        text = (r.get("title") or "") + " " + (r.get("content") or "")
        s = _parse_settlement(text)
        if not s:
            continue
        return {
            "key":      metal["key"],
            "label":    metal["label"],
            "level":    s.get("level"),
            "pct":      s.get("pct"),
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
    for metal in METALS:
        try:
            point = await asyncio.wait_for(_fetch_one(metal), timeout=15.0)
        except asyncio.TimeoutError:
            log.info("[lme_metals:%s] per-metal timeout", metal["key"])
            continue
        if not point:
            continue

        # Always log the datapoint to metrics_daily so the chart surface
        # has continuity, even when the move is below the firing threshold.
        try:
            sb.upsert_metric_daily(
                metric_key=f"metals_{metal['key']}",
                captured_at=captured,
                value_num=point.get("level"),
                source=f"lme_metals:{metal['key']}",
                bucket="construction",
                unit=metal["unit"],
                raw_json={
                    "pct_dod":   point.get("pct"),
                    "url":       point.get("url"),
                    "title":     point.get("title"),
                    "label":     metal["label"],
                },
            )
        except Exception:
            pass

        # Fire as signal only when daily move is material.
        pct = point.get("pct")
        if pct is None or abs(pct) < metal["fire_threshold_pct"]:
            continue

        arrow = "↑" if pct > 0 else "↓"
        title = (
            f"Metals · {metal['label']} {arrow}{abs(pct):.1f}% DoD"
            + (f" to {point['level']:,.0f} {metal['unit']}" if point.get("level") else "")
        )
        summary = (
            f"{metal['label']} moved {arrow}{abs(pct):.1f}% day-over-day"
            + (f" to {point['level']:,.0f} {metal['unit']}." if point.get("level") else ".") + " "
            f"Direct read on Sobha gross-margin pressure — steel + copper + "
            f"aluminum drive ~30% of tower-construction input cost. "
            f"Source: {point.get('title', '')}."
        )
        dedup_key = hashlib.sha256(
            f"lme_metals:{metal['key']}:{captured.date().isoformat()}".encode()
        ).hexdigest()
        out.append({
            "source":     f"lme_metals:{metal['key']}",
            "source_url": point.get("url") or "",
            "title":      title[:300],
            "summary":    summary[:600],
            "raw_json": {
                "metric_key":    metal["key"],
                "label":         metal["label"],
                "level":         point.get("level"),
                "pct_dod":       pct,
                "unit":          metal["unit"],
                "category_hint": "materials",
                "tier_hint":     "press",
                "region":        "other",
                "country_code":  None,
            },
            "dedup_key": dedup_key,
        })
        if len(out) >= limit:
            break

    log.info("[lme_metals] metals=%d fired=%d", len(METALS), len(out))
    return out
