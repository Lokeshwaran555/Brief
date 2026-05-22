"""BDI + SCFI shipping leading-indicator scout (Round-2, 2026-04-29).

Drewry's WCI (already covered by materials_scout) is the de-facto
container-shipping benchmark, but it lags. BDI (Baltic Dry Index — bulk
materials, mostly Capesize / Panamax / Supramax) and SCFI (Shanghai
Containerized Freight Index — Asia-out container rates) move 30-60 days
AHEAD of Drewry as leading indicators of supply-chain pressure that
will eventually hit UAE landed material costs.

We pull weekly index moves (% change) and stamp them into the
metrics_daily time-series store + emit a signal when the WoW move is
material (≥5% for BDI, ≥3% for SCFI given different volatility regimes).

Stamp:
  - source: "bdi_scfi:bdi" / "bdi_scfi:scfi"
  - category_hint: "materials"
  - tier_hint: "press" (aggregator-published values)
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


INDICES: list[dict[str, Any]] = [
    {
        "key":        "bdi",
        "label":      "Baltic Dry Index",
        "query":      "Baltic Dry Index BDI weekly value point change",
        "domains":    ["balticexchange.com", "hellenicshippingnews.com",
                       "splash247.com", "container-news.com",
                       "tradingeconomics.com", "reuters.com",
                       "bloomberg.com", "ft.com"],
        "fire_threshold_pct": 5.0,   # ≥5% WoW move → fire as signal
        "rough_baseline":     1500.0, # for unit context only
    },
    {
        "key":        "scfi",
        "label":      "Shanghai Containerized Freight Index",
        "query":      "Shanghai Containerized Freight Index SCFI weekly point change USD container",
        "domains":    ["en.sse.net.cn", "hellenicshippingnews.com",
                       "splash247.com", "container-news.com",
                       "joc.com", "loadstar.co.uk",
                       "tradingeconomics.com", "reuters.com",
                       "bloomberg.com"],
        "fire_threshold_pct": 3.0,   # SCFI is more volatile; 3% is meaningful
        "rough_baseline":     1300.0,
    },
]


# Try to extract a current value + WoW change from headlines.
# Examples:
#   "BDI rose 4.5% to 1,742"
#   "SCFI fell 2.1 percent to 1,234"
#   "Baltic Dry Index gained 87 points to 1,742"
_PCT_RX = re.compile(
    r"(rose|fell|gained|lost|up|down|jumped|slipped|climbed|dropped)[^\d]{0,20}"
    r"(\d{1,3}(?:\.\d+)?)\s*(%|percent|pct|points?)",
    re.IGNORECASE,
)
_LEVEL_RX = re.compile(r"\b(?:to|at|reached|stands at)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def _parse_move(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = _PCT_RX.search(text)
    if not m:
        return None
    word, num, unit = m.group(1).lower(), m.group(2), m.group(3).lower()
    try:
        v = float(num)
    except Exception:
        return None
    direction = -1 if word in ("fell", "lost", "down", "slipped", "dropped") else +1
    pct: float | None = None
    points: float | None = None
    if "%" in unit or "percent" in unit or "pct" in unit:
        pct = v * direction
    else:
        points = v * direction
    # Level (current value) extraction — best effort.
    level: float | None = None
    lm = _LEVEL_RX.search(text)
    if lm:
        try:
            level = float(lm.group(1).replace(",", ""))
        except Exception:
            pass
    return {"pct": pct, "points": points, "level": level}


async def _fetch_one(index: dict[str, Any]) -> dict[str, Any] | None:
    try:
        results = await tavily_search.search(
            index["query"],
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=index["domains"],
            days=14,
        )
    except Exception as e:
        log.warning("[bdi_scfi:%s] tavily failed: %s", index["key"], e)
        return None
    for r in results or []:
        text = (r.get("title") or "") + " " + (r.get("content") or "")
        move = _parse_move(text)
        if not move:
            continue
        return {
            "key":      index["key"],
            "label":    index["label"],
            "pct":      move.get("pct"),
            "points":   move.get("points"),
            "level":    move.get("level"),
            "url":      r.get("url"),
            "title":    (r.get("title") or "")[:300],
            "snippet":  text[:600],
            "published": r.get("published_date"),
        }
    return None


async def run(limit: int = 4) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    captured = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for idx in INDICES:
        point = await _fetch_one(idx)
        if not point:
            continue

        # Time-series log (always, for chart continuity).
        try:
            sb.upsert_metric_daily(
                metric_key=f"shipping_{idx['key']}",
                captured_at=captured,
                value_num=point.get("level") or idx["rough_baseline"],
                source=f"bdi_scfi:{idx['key']}",
                bucket="construction",
                unit="index",
                raw_json={
                    "pct_wow":  point.get("pct"),
                    "pts_wow":  point.get("points"),
                    "url":      point.get("url"),
                    "title":    point.get("title"),
                },
            )
        except Exception:
            pass

        # Fire as signal only when WoW move is material.
        pct = point.get("pct")
        if pct is None or abs(pct) < idx["fire_threshold_pct"]:
            continue

        arrow = "↑" if pct > 0 else "↓"
        title = (
            f"Shipping · {idx['label']} {arrow}{abs(pct):.1f}% WoW"
            + (f" to {point['level']:,.0f}" if point.get("level") else "")
        )
        summary = (
            f"{idx['label']} ({idx['key'].upper()}) moved {arrow}{abs(pct):.1f}% week-on-week"
            + (f" to {point['level']:,.0f}." if point.get("level") else ".") + " "
            f"Leads Drewry's WCI by 30-60 days — UAE landed-cost pressure incoming. "
            f"Source: {point.get('title', '')}."
        )
        dedup_key = hashlib.sha256(
            f"bdi_scfi:{idx['key']}:{captured.date().isoformat()}".encode()
        ).hexdigest()
        out.append({
            "source":     f"bdi_scfi:{idx['key']}",
            "source_url": point.get("url") or "",
            "title":      title[:300],
            "summary":    summary[:600],
            "raw_json": {
                "index_key":     idx["key"],
                "label":         idx["label"],
                "pct_wow":       pct,
                "pts_wow":       point.get("points"),
                "level":         point.get("level"),
                "category_hint": "materials",
                "tier_hint":     "press",
                "region":        "other",
                "country_code":  None,
            },
            "dedup_key": dedup_key,
        })
        if len(out) >= limit:
            break

    log.info("[bdi_scfi] indices=%d fired=%d", len(INDICES), len(out))
    return out
