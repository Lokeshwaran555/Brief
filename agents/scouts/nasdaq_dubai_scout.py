"""NASDAQ Dubai sukuk + bond daily scout (Round-3, 2026-04-29).

Replaces sukuk_scout's Tavily-inference path for the issuers listed on
NASDAQ Dubai. Extracts the headline daily yield + price change for:
  - Sobha Sukuk 2030
  - Damac Sukuk
  - Emaar Sukuk
  - Aldar Sukuk
  - UAE sovereign sukuk (DXBSUK)
  - DP World, Etisalat, Mashreq sukuks (broader UAE benchmark stack)

NASDAQ Dubai publishes daily-close bond prices on nasdaqdubai.com but
the markup shifts; Tavily-mediated extraction with strict domain whitelist
is the reliable path. Yields go to metrics_daily, price moves > 50bps
fire as signals.

Stamp:
  - source: "nasdaq_dubai:<issuer>"
  - category_hint: "capital_markets"
  - tier_hint: "official"
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


ISSUERS: list[dict[str, Any]] = [
    {
        "key":     "sobha_sukuk_2030",
        "label":   "Sobha Sukuk 2030",
        "query":   "Sobha sukuk 2030 yield price daily Nasdaq Dubai close",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "bloomberg.com", "reuters.com", "ft.com",
                    "spglobal.com", "moodys.com", "fitchratings.com"],
    },
    {
        "key":     "damac_sukuk",
        "label":   "Damac Sukuk",
        "query":   "DAMAC sukuk yield price daily Nasdaq Dubai close 2026",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "thenationalnews.com", "bloomberg.com", "reuters.com",
                    "ft.com", "spglobal.com"],
    },
    {
        "key":     "emaar_sukuk",
        "label":   "Emaar Sukuk",
        "query":   "Emaar sukuk yield price daily Nasdaq Dubai close 2026",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "thenationalnews.com", "bloomberg.com", "reuters.com",
                    "ft.com"],
    },
    {
        "key":     "aldar_sukuk",
        "label":   "Aldar Sukuk",
        "query":   "Aldar sukuk yield price daily Nasdaq Dubai close 2026",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "thenationalnews.com", "bloomberg.com", "reuters.com",
                    "ft.com"],
    },
    {
        "key":     "uae_sovereign",
        "label":   "UAE Sovereign Sukuk",
        "query":   "UAE sovereign sukuk yield Nasdaq Dubai daily close 2026",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "thenationalnews.com", "bloomberg.com", "reuters.com",
                    "ft.com", "spglobal.com"],
    },
    {
        "key":     "abu_dhabi_sovereign",
        "label":   "Abu Dhabi Sovereign Sukuk",
        "query":   "Abu Dhabi sovereign sukuk yield daily close 2026",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "thenationalnews.com", "bloomberg.com", "reuters.com",
                    "ft.com"],
    },
]


# Patterns: "Sobha sukuk yield 5.85%", "Aldar sukuk traded at 102.45 yielding 4.95%"
_YIELD_RX = re.compile(
    r"(?:yield(?:ing)?|coupon|return)\s*(?:at|of|to|stood at)?\s*"
    r"([\d.]+)\s*%",
    re.IGNORECASE,
)
_PRICE_RX = re.compile(
    r"(?:traded at|trading at|priced at|closed at|stood at)\s*"
    r"\$?([\d.]+)",
    re.IGNORECASE,
)
_BPS_RX = re.compile(
    r"(tightened|widened|narrowed|fell|rose)\s*(\d{1,3})\s*(bps|basis points)",
    re.IGNORECASE,
)


def _parse_yield(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    out: dict[str, Any] = {}
    ym = _YIELD_RX.search(text)
    if ym:
        try:
            out["yield_pct"] = float(ym.group(1))
        except Exception:
            pass
    pm = _PRICE_RX.search(text)
    if pm:
        try:
            out["price"] = float(pm.group(1))
        except Exception:
            pass
    bm = _BPS_RX.search(text)
    if bm:
        word = bm.group(1).lower()
        try:
            bps = int(bm.group(2))
        except Exception:
            bps = 0
        sign = -1 if word in ("tightened", "narrowed", "fell") else +1
        out["bps_move"] = sign * bps
    return out or None


async def _fetch_one(issuer: dict[str, Any]) -> dict[str, Any] | None:
    try:
        results = await tavily_search.search(
            issuer["query"], max_results=4,
            topic="news", search_depth="basic",
            include_domains=issuer["domains"],
            days=14,
        )
    except Exception as e:
        log.warning("[nasdaq_dubai:%s] tavily failed: %s", issuer["key"], e)
        return None
    for r in results or []:
        text = (r.get("title") or "") + " " + (r.get("content") or "")
        d = _parse_yield(text)
        if not d:
            continue
        return {
            "key":       issuer["key"],
            "label":     issuer["label"],
            "yield_pct": d.get("yield_pct"),
            "price":     d.get("price"),
            "bps_move":  d.get("bps_move"),
            "url":       r.get("url"),
            "title":     (r.get("title") or "")[:300],
            "snippet":   text[:600],
            "published": r.get("published_date"),
        }
    return None


async def run(limit: int = 8) -> list[dict[str, Any]]:
    captured = datetime.now(timezone.utc).replace(hour=0, minute=0,
                                                   second=0, microsecond=0)
    out: list[dict[str, Any]] = []
    for issuer in ISSUERS:
        try:
            point = await asyncio.wait_for(_fetch_one(issuer), timeout=15.0)
        except asyncio.TimeoutError:
            continue
        if not point:
            continue

        # Log yield to metrics_daily for chart continuity.
        if point.get("yield_pct") is not None:
            try:
                sb.upsert_metric_daily(
                    metric_key=f"sukuk_{issuer['key']}_yield",
                    captured_at=captured,
                    value_num=point["yield_pct"],
                    source=f"nasdaq_dubai:{issuer['key']}",
                    bucket="markets",
                    unit="%",
                    raw_json={
                        "price":     point.get("price"),
                        "bps_move":  point.get("bps_move"),
                        "url":       point.get("url"),
                        "title":     point.get("title"),
                    },
                )
            except Exception:
                pass

        # Fire as signal when bps move ≥50bps OR yield change is material.
        bps = point.get("bps_move") or 0
        if abs(bps) < 50:
            continue

        arrow = "↑" if bps > 0 else "↓"
        title = (
            f"Sukuk · {issuer['label']} yield {arrow}{abs(bps)}bps"
            + (f" to {point['yield_pct']:.2f}%" if point.get("yield_pct") else "")
        )
        summary = (
            f"{issuer['label']} {arrow}{abs(bps)}bps in the last session"
            + (f" to {point['yield_pct']:.2f}% yield." if point.get("yield_pct") else ".") + " "
            f"Capital-market read on UAE developer credit risk. "
            f"Source: {point.get('title', '')}."
        )
        dedup_key = hashlib.sha256(
            f"nasdaq_dubai:{issuer['key']}:{captured.date().isoformat()}".encode()
        ).hexdigest()
        out.append({
            "source":     f"nasdaq_dubai:{issuer['key']}",
            "source_url": point.get("url") or "https://www.nasdaqdubai.com/",
            "title":      title[:300],
            "summary":    summary[:600],
            "raw_json": {
                "metric_key":    issuer["key"],
                "label":         issuer["label"],
                "yield_pct":     point.get("yield_pct"),
                "price":         point.get("price"),
                "bps_move":      bps,
                "category_hint": "capital_markets",
                "tier_hint":     "official",
                "region":        "dubai",
                "country_code":  "AE",
            },
            "dedup_key": dedup_key,
        })
        if len(out) >= limit:
            break

    log.info("[nasdaq_dubai] issuers=%d fired=%d", len(ISSUERS), len(out))
    return out
