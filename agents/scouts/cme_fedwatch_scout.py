"""CME FedWatch scout — implied probability of next Fed rate move.

The AED is pegged to USD, so EIBOR follows the Fed funds rate almost
1:1 over the medium term. CME's FedWatch tool publishes the market-
implied probability distribution for the next FOMC meeting based on
30-Day Fed Fund futures pricing. One number, very dense signal.

We pull the headline probability of the most-likely next move (cut /
hold / hike) and store it in metrics_daily for the chart surface, plus
emit a signal when the headline probability shifts ≥10 percentage
points day-over-day (which is the threshold where positioning has
materially changed).

Sources tried (fail-soft chain):
  1. CME FedWatch directly — undocumented JSON endpoint
  2. Tavily-mediated query against cmegroup.com + financial press
     (Bloomberg, Reuters, FT) for the headline number

Stamp:
  - source: "cme_fedwatch:<meeting_iso>"
  - category_hint: "capital_markets"
  - tier_hint: "official" (CME is an SRO publishing futures data)
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


# Fall back to Tavily when the direct endpoint fails. Trusted publishers
# that report FedWatch state in headlines.
DOMAINS = [
    "cmegroup.com", "bloomberg.com", "reuters.com", "ft.com", "wsj.com",
    "marketwatch.com", "cnbc.com", "investopedia.com",
]


_PROB_RX = re.compile(
    r"(?:probability|odds|chance)[^\d]{0,30}([0-9]{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_DIRECTION_RX = re.compile(r"\b(rate\s*cut|cut|hold|pause|hike|raise|increase)\b", re.IGNORECASE)


def _direction_from_text(text: str) -> str:
    """Pick the dominant move-direction word from the text."""
    if not text:
        return "hold"
    t = text.lower()
    if "cut" in t or "lower" in t:
        return "cut"
    if "hike" in t or "raise" in t or "increase" in t:
        return "hike"
    return "hold"


async def _via_tavily() -> dict[str, Any] | None:
    """Best-effort Tavily extraction of the headline FedWatch probability."""
    query = "CME FedWatch probability of Fed rate cut hold hike next FOMC meeting"
    try:
        results = await tavily_search.search(
            query,
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=DOMAINS,
            days=7,
        )
    except Exception as e:
        log.warning("[cme_fedwatch] tavily failed: %s", e)
        return None
    for r in results or []:
        text = (r.get("content") or "") + " " + (r.get("title") or "")
        m = _PROB_RX.search(text)
        if not m:
            continue
        try:
            prob = float(m.group(1))
        except Exception:
            continue
        if prob < 0 or prob > 100:
            continue
        return {
            "probability": prob,
            "direction":   _direction_from_text(text),
            "url":         r.get("url"),
            "title":       (r.get("title") or "")[:300],
            "snippet":     text[:600],
            "published":   r.get("published_date"),
        }
    return None


async def run(limit: int = 3) -> list[dict[str, Any]]:
    point = await _via_tavily()
    if not point:
        log.info("[cme_fedwatch] no headline probability extracted")
        return []

    # Always log the datapoint regardless of delta, so the chart surface
    # has continuity. metric_key includes direction so cut/hold/hike each
    # get their own series.
    direction = point["direction"]
    captured = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        sb.upsert_metric_daily(
            metric_key=f"cme_fedwatch_{direction}_prob",
            captured_at=captured,
            value_num=point["probability"],
            source="cme_fedwatch",
            bucket="markets",
            unit="%",
            raw_json={
                "direction": direction,
                "url":       point.get("url"),
                "title":     point.get("title"),
            },
        )
    except Exception:
        pass

    # Day-over-day delta detection. Pull yesterday's value from
    # metrics_daily; if delta ≥ 10pp, fire as a signal.
    fire = False
    delta_pp = 0.0
    try:
        prior = (
            sb.client().table("metrics_daily")
            .select("value_num,captured_at")
            .eq("metric_key", f"cme_fedwatch_{direction}_prob")
            .lt("captured_at", captured.isoformat())
            .order("captured_at", desc=True).limit(1).execute().data or []
        )
        if prior:
            try:
                prev_val = float(prior[0].get("value_num") or 0)
                delta_pp = point["probability"] - prev_val
                if abs(delta_pp) >= 10.0:
                    fire = True
            except Exception:
                pass
    except Exception:
        pass

    if not fire:
        log.info("[cme_fedwatch] %s prob=%.1f%% (Δ%.1fpp) — below firing threshold",
                 direction, point["probability"], delta_pp)
        return []

    arrow = "↑" if delta_pp > 0 else "↓"
    title = (
        f"CME FedWatch · {direction} probability {arrow}{abs(delta_pp):.0f}pp "
        f"to {point['probability']:.0f}% (next FOMC)"
    )
    summary = (
        f"Implied probability of a Fed {direction} at the next FOMC meeting "
        f"shifted {arrow}{abs(delta_pp):.1f}pp day-over-day to {point['probability']:.1f}%. "
        f"Drives EIBOR via the AED/USD peg — Sobha mortgage-affordability lens. "
        f"Source: {point.get('title', 'CME / financial press')}."
    )
    dedup_key = hashlib.sha256(
        f"cme_fedwatch:{direction}:{captured.date().isoformat()}".encode()
    ).hexdigest()
    return [{
        "source":     "cme_fedwatch:next_fomc",
        "source_url": point.get("url") or "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        "title":      title[:300],
        "summary":    summary[:600],
        "raw_json": {
            "probability_pct": point["probability"],
            "direction":       direction,
            "delta_pp":        delta_pp,
            "category_hint":   "capital_markets",
            "tier_hint":       "official",
            "region":          "other",
            "country_code":    "US",
        },
        "dedup_key": dedup_key,
    }][:limit]
