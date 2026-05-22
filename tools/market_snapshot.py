"""Lightweight market snapshot — Brent, S&P, gold, DXY, INR FX, AED peg.

Used by:
  1. flows/daily_brief_flow.py — to inject a 1-line market lead at the
     top of every brief
  2. server.py:/api/friday-chat — to give Friday the same market context
     when boss asks anything market-adjacent

Sources:
  - Yahoo Finance unofficial v8 chart API for quotes (free, no key)
  - AED is pegged to USD at 3.6725 (UAE Central Bank); static.

Defensive: every fetch is best-effort. Failure → element omitted from
snapshot rather than blowing up the brief.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)


# Yahoo Finance v8 chart API — returns most-recent close + previous close.
YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2d&interval=1d"

# Tickers we read each call.
# Categorized so the dashboard "Markets & Capital" page can group them
# (indices, rates, FX, commodities, MAG-7 strip, UAE listed RE).
# Keep the pre-2026-04-28 set first so the brief one-liner stays stable.
TICKERS: list[dict[str, str]] = [
    # Headline indices + commodities (existing — keep order, brief one-liner reads first 5)
    {"ticker": "^GSPC",     "label": "S&P 500",     "group": "index"},
    {"ticker": "BZ=F",      "label": "Brent",       "group": "commodity"},
    {"ticker": "GC=F",      "label": "Gold",        "group": "commodity"},
    {"ticker": "DX-Y.NYB",  "label": "DXY",         "group": "fx"},
    {"ticker": "INR=X",     "label": "USD/INR",     "group": "fx"},
    {"ticker": "BTC-USD",   "label": "BTC",         "group": "crypto"},
    # 2026-04-28 stakeholder ask: Markets & Capital page additions
    # ─── US indices (added 2026-05-07: stakeholder feedback that
    # Nasdaq + Dow values weren't surfacing on the Markets page) ───
    {"ticker": "^IXIC",     "label": "Nasdaq",      "group": "index"},
    {"ticker": "^DJI",      "label": "Dow",         "group": "index"},
    {"ticker": "^RUT",      "label": "Russell 2000","group": "index"},
    {"ticker": "^VIX",      "label": "VIX",         "group": "index"},
    # ─── US rates ───
    {"ticker": "^TNX",      "label": "US 10Y",      "group": "rate"},   # 10-year Treasury yield ×10
    {"ticker": "^IRX",      "label": "US 3M",       "group": "rate"},   # 13-week T-bill (proxy for short end)
    {"ticker": "^FVX",      "label": "US 5Y",       "group": "rate"},
    # ─── Commodities (extension) ───
    {"ticker": "CL=F",      "label": "WTI",         "group": "commodity"},  # WTI crude
    {"ticker": "ETH-USD",   "label": "ETH",         "group": "crypto"},
    {"ticker": "HG=F",      "label": "Copper",      "group": "commodity"},  # Copper futures (LME proxy)
    # ─── FX (vs USD; AED = USD × 3.6725) ───
    {"ticker": "GBP=X",     "label": "USD/GBP",     "group": "fx"},
    {"ticker": "EUR=X",     "label": "USD/EUR",     "group": "fx"},
    {"ticker": "AUDUSD=X",  "label": "AUD/USD",     "group": "fx"},
    {"ticker": "CNY=X",     "label": "USD/CNY",     "group": "fx"},
    {"ticker": "RUB=X",     "label": "USD/RUB",     "group": "fx"},
    # ─── MAG-7 strip ───
    {"ticker": "AAPL",      "label": "Apple",       "group": "mag7"},
    {"ticker": "MSFT",      "label": "Microsoft",   "group": "mag7"},
    {"ticker": "GOOGL",     "label": "Alphabet",    "group": "mag7"},
    {"ticker": "AMZN",      "label": "Amazon",      "group": "mag7"},
    {"ticker": "META",      "label": "Meta",        "group": "mag7"},
    {"ticker": "NVDA",      "label": "Nvidia",      "group": "mag7"},
    {"ticker": "TSLA",      "label": "Tesla",       "group": "mag7"},
    # ─── UAE listed RE (DFM/ADX via Yahoo) ───
    # Suffixes: .AE = ADX (Aldar), .DU = DFM (Emaar, Damac, Deyaar, Union).
    # Yahoo coverage is best-effort; missing rows fail soft per _fetch_one.
    {"ticker": "EMAAR.DU",  "label": "Emaar",       "group": "uae_re"},
    {"ticker": "DAMAC.DU",  "label": "Damac",       "group": "uae_re"},
    {"ticker": "ALDAR.AE",  "label": "Aldar",       "group": "uae_re"},
    {"ticker": "DEYAAR.DU", "label": "Deyaar",      "group": "uae_re"},
    {"ticker": "UPP.DU",    "label": "Union Props", "group": "uae_re"},
    {"ticker": "EMIRATESREIT.DU", "label": "Emirates REIT", "group": "uae_re"},
    # ─── Construction commodity / equity proxies (Yahoo, no login) ───
    # Dedicated proxies for the Construction Economics page. Equity
    # proxies (SLX, CPER, VALE, AA, CX, EXP, WY) are reliably free on
    # Yahoo whereas continuous-contract futures (HRC1!, FEF1!, ALI1!,
    # LBR1!) often need TradingView login. These give us the same
    # signal without the auth ceiling.
    {"ticker": "SLX",       "label": "Steel ETF (SLX)",     "group": "construction"},
    {"ticker": "CPER",      "label": "Copper ETF (CPER)",   "group": "construction"},
    {"ticker": "VALE",      "label": "Vale (iron ore)",     "group": "construction"},
    {"ticker": "AA",        "label": "Alcoa (aluminum)",    "group": "construction"},
    {"ticker": "CX",        "label": "Cemex (cement)",      "group": "construction"},
    {"ticker": "EXP",       "label": "Eagle Materials",     "group": "construction"},
    {"ticker": "WY",        "label": "Weyerhaeuser (lumber)","group": "construction"},
    {"ticker": "FCX",       "label": "Freeport (copper)",   "group": "construction"},
    {"ticker": "NUE",       "label": "Nucor (steel)",       "group": "construction"},
    {"ticker": "URI",       "label": "United Rentals",      "group": "construction"},
    # ─── Container shipping equity (BDI/SCFI proxy) ───
    {"ticker": "ZIM",       "label": "ZIM Shipping",        "group": "shipping"},
    {"ticker": "DAC",       "label": "Danaos",              "group": "shipping"},
    {"ticker": "GSL",       "label": "Global Ship Lease",   "group": "shipping"},
    {"ticker": "CMRE",      "label": "Costamare",           "group": "shipping"},
    {"ticker": "STNG",      "label": "Scorpio Tankers",     "group": "shipping"},
]


# 2026-04-29: Yahoo v8 is unofficial and breaks periodically. When it
# returns garbage / 429s for a ticker, we fall through to Twelve Data
# (free tier ~800 calls/day) for a quote on the same symbol.
# TWELVEDATA_API_KEY is optional — when unset, we just degrade more
# gracefully on Yahoo failures (return None instead of fabricated data).
_TWELVE_QUOTE_URL = "https://api.twelvedata.com/quote?symbol={sym}&apikey={key}"

# Map of our internal label → the symbol Twelve Data uses. Twelve Data
# accepts most Yahoo-style tickers as-is (AAPL, ^GSPC, BTC/USD), but a
# few of our forex / index symbols need a translation.
_TWELVE_SYMBOL_MAP: dict[str, str] = {
    "DXY":     "DXY",
    "S&P 500": "SPX",
    "Nasdaq":  "IXIC",
    "Dow":     "DJI",
    "Russell 2000": "RUT",
    "VIX":     "VIX",
    "Brent":   "BRENT",
    "Gold":    "XAU/USD",
    "WTI":     "WTI",
    "BTC":     "BTC/USD",
    "ETH":     "ETH/USD",
    "Copper":  "HG=F",
    "USD/INR": "USD/INR",
    "USD/GBP": "USD/GBP",
    "USD/EUR": "USD/EUR",
    "USD/CNY": "USD/CNY",
    "USD/RUB": "USD/RUB",
    "AUD/USD": "AUD/USD",
    "US 10Y":  "TNX",
    "US 5Y":   "FVX",
    "US 3M":   "IRX",
    # ── UAE listed RE (added 2026-04-30: Yahoo stopped covering
    # .DU/.AE suffixes; Twelve Data picks up the slack via DFM/ADX
    # exchange suffixes). Tried the bare symbol form which Twelve Data
    # disambiguates by exchange parameter — see _fetch_twelvedata.
    "Emaar":         "EMAAR",
    "Damac":         "DAMAC",
    "Aldar":         "ALDARPJSC",
    "Deyaar":        "DEYAAR",
    "Union Props":   "UPP",
    "Emirates REIT": "REIT",
}


# UAE-listed RE labels need an exchange parameter on Twelve Data
# because the bare symbols collide with US OTC tickers.
_TWELVE_UAE_EXCHANGE: dict[str, str] = {
    "Emaar":         "DFM",   # Dubai Financial Market
    "Damac":         "DFM",
    "Aldar":         "ADX",   # Abu Dhabi Securities Exchange
    "Deyaar":        "DFM",
    "Union Props":   "DFM",
    "Emirates REIT": "DFM",
}


# Multiple ticker forms per UAE label — Twelve Data's symbol naming
# is inconsistent for DFM/ADX (sometimes EMAAR, sometimes EMAARDEV;
# Aldar may be ALDAR or ALDARPJSC). We try forms in order; first one
# that returns data wins.
_TWELVE_UAE_SYMBOL_VARIANTS: dict[str, list[str]] = {
    "Emaar":         ["EMAAR", "EMAARDEV", "EMAARPROP"],
    "Damac":         ["DAMAC", "DAMACPROP"],
    "Aldar":         ["ALDAR", "ALDARPJSC", "ALDARRE"],
    "Deyaar":        ["DEYAAR", "DEYAARRE"],
    "Union Props":   ["UPP", "UNIONPROP", "UPPS"],
    "Emirates REIT": ["REIT", "EMREIT", "EMIRATESREIT"],
}


async def _fetch_twelvedata(client: httpx.AsyncClient, t: dict[str, str]) -> dict[str, Any] | None:
    """Fallback quote via Twelve Data when Yahoo returns nothing.

    UAE-listed labels (Emaar/Damac/Aldar/Deyaar/Union Props/Emirates
    REIT) need an `exchange=DFM|ADX` parameter because their bare
    symbols collide with US OTC tickers. We also try multiple symbol
    variants (EMAAR / EMAARDEV / EMAARPROP) since Twelve Data's UAE
    naming is inconsistent — first variant that returns valid data
    wins. Non-UAE labels skip both extras and use the bare ticker.
    """
    import os as _os
    key = (_os.environ.get("TWELVEDATA_API_KEY") or "").strip()
    if not key:
        return None
    exchange = _TWELVE_UAE_EXCHANGE.get(t["label"])
    # Build symbol-try list: UAE labels get all variants, others just the mapped/raw ticker
    if exchange and t["label"] in _TWELVE_UAE_SYMBOL_VARIANTS:
        symbols_to_try = _TWELVE_UAE_SYMBOL_VARIANTS[t["label"]]
    else:
        symbols_to_try = [_TWELVE_SYMBOL_MAP.get(t["label"], t["ticker"])]
    data = None
    for sym in symbols_to_try:
        url = _TWELVE_QUOTE_URL.format(sym=sym, key=key)
        if exchange:
            url += f"&exchange={exchange}"
        try:
            resp = await client.get(url, timeout=8.0)
            resp.raise_for_status()
            d = resp.json()
        except Exception as e:
            log.info("[market:twelvedata] %s/%s fetch failed: %s",
                     t["ticker"], sym, type(e).__name__)
            continue
        # Twelve Data returns {"status":"error", ...} or {"code": 404}
        # when the symbol is unknown. Move on to next variant.
        if d.get("status") == "error" or d.get("code") == 404:
            continue
        if not d.get("close"):
            continue
        log.info("[market:twelvedata] %s resolved via symbol=%s exchange=%s",
                 t["label"], sym, exchange or "-")
        data = d
        break
    if not data:
        return None
    try:
        if data.get("status") == "error" or not data.get("close"):
            return None
        last = float(data["close"])
        prev = float(data.get("previous_close") or 0)
        if not last:
            return None
        change_pct = ((last - prev) / prev * 100.0) if prev else None
        return {
            "label":      t["label"],
            "ticker":     t["ticker"],
            "group":      t.get("group", "index"),
            "value":      last,
            "prev_close": prev,
            "change_pct": change_pct,
            "_source":    "twelvedata",  # provenance for debugging
        }
    except Exception as e:
        log.info("[market:twelvedata] %s parse failed: %s", t["ticker"], e)
        return None


async def _fetch_one(client: httpx.AsyncClient, t: dict[str, str]) -> dict[str, Any] | None:
    """Yahoo first, Twelve Data as fallback when Yahoo returns nothing."""
    try:
        resp = await client.get(
            YF_URL.format(ticker=t["ticker"]),
            headers={"User-Agent": "Sobha MDI market snapshot (contact: ops@sobha.com)"},
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[market] %s yahoo fetch failed: %s — trying fallback",
                    t["ticker"], type(e).__name__)
        return await _fetch_twelvedata(client, t)

    try:
        result = (data.get("chart") or {}).get("result") or []
        if not result:
            return await _fetch_twelvedata(client, t)
        meta = result[0].get("meta") or {}
        prev = float(meta.get("chartPreviousClose") or meta.get("previousClose") or 0)
        last = float(meta.get("regularMarketPrice") or 0)
        if not last:
            ind = (result[0].get("indicators") or {}).get("quote") or [{}]
            closes = ind[0].get("close") or []
            for c in reversed(closes):
                if c is not None:
                    last = float(c)
                    break
        if not last:
            return await _fetch_twelvedata(client, t)
        change_pct = ((last - prev) / prev * 100.0) if prev else None
        # Best-effort write to metrics_daily for the chart surface.
        # Doesn't block snapshot emission if the table doesn't exist.
        try:
            from tools import supabase_tool as _sb
            from datetime import datetime as _dt
            _sb.upsert_metric_daily(
                metric_key=f"yf_{t['ticker'].replace('=', '_').replace('^','idx_').replace('-','_')}",
                captured_at=_dt.now(timezone.utc),
                value_num=last,
                source="yahoo:close",
                bucket={"index":"markets","rate":"markets","fx":"markets",
                        "commodity":"construction","crypto":"markets",
                        "mag7":"markets","uae_re":"markets"}.get(t.get("group"), "markets"),
                unit="quote",
                raw_json={"label": t["label"], "prev": prev, "change_pct": change_pct},
            )
        except Exception:
            pass
        return {
            "label":      t["label"],
            "ticker":     t["ticker"],
            "group":      t.get("group", "index"),
            "value":      last,
            "prev_close": prev,
            "change_pct": change_pct,
            "_source":    "yahoo",
        }
    except Exception as e:
        log.warning("[market] %s yahoo parse failed: %s — trying fallback",
                    t["ticker"], e)
        return await _fetch_twelvedata(client, t)


def _format_value(label: str, value: float) -> str:
    """Human-readable value per asset class."""
    if label in ("S&P 500", "Nasdaq", "Dow", "Russell 2000"):
        return f"{value:,.0f}"
    if label == "VIX":
        return f"{value:.2f}"
    if label in ("Brent", "Gold", "WTI"):
        return f"${value:,.1f}"
    if label == "DXY":
        return f"{value:.1f}"
    if label == "USD/INR":
        return f"₹{value:.2f}"
    if label in ("BTC", "ETH"):
        return f"${value:,.0f}"
    # Yahoo's ^TNX/^IRX/^FVX are yields ×10 (e.g. 4.5% comes back as 45).
    if label in ("US 10Y", "US 5Y", "US 3M"):
        return f"{value / 10:.2f}%"
    if label == "Copper":
        return f"${value:,.2f}"
    if label.startswith("USD/") or label.endswith("/USD"):
        return f"{value:.4f}" if value < 10 else f"{value:.2f}"
    # Construction / shipping equity proxies — show with $ prefix.
    # MAG-7 + UAE listed RE (no prefix, native quote).
    return f"{value:,.2f}"


def _format_pct(p: float | None) -> str:
    if p is None:
        return "—"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def _build_one_liner(rows: list[dict[str, Any]]) -> str:
    """Tight one-line market lead for the brief writer to drop in.

    Skips elements whose change is null. Always ends with the AED peg
    note so the MD has the day's FX backdrop in one glance.

    Lead with the 5 brief-stable rows (S&P, Brent, Gold, DXY, USD/INR) —
    the MAG-7 + UAE-RE strip is for the Markets & Capital page, not
    the morning-brief one-liner.
    """
    LEAD_LABELS = ("S&P 500", "Brent", "Gold", "DXY", "USD/INR")
    bits: list[str] = []
    for label in LEAD_LABELS:
        r = next((x for x in rows if x.get("label") == label), None)
        if not r or r.get("change_pct") is None:
            continue
        v = _format_value(r["label"], r["value"])
        c = _format_pct(r["change_pct"])
        bits.append(f"{r['label']} {v} ({c})")
    if not bits:
        return ""
    line = ", ".join(bits)
    return f"Markets: {line}. AED steady at 3.6725 (USD peg)."


def grouped_for_dashboard(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket rows by `group` for the Markets & Capital page.

    Buckets: index, rate, fx, commodity, crypto, mag7, uae_re.
    Stable order so the dashboard renders consistently.
    """
    buckets: dict[str, list[dict[str, Any]]] = {
        "index":     [],
        "rate":      [],
        "fx":        [],
        "commodity": [],
        "crypto":    [],
        "mag7":      [],
        "uae_re":    [],
    }
    for r in rows:
        g = r.get("group") or "index"
        buckets.setdefault(g, []).append(r)
    return buckets


# 2026-04-30 audit-round: 5-min TTL process-local cache. Without this,
# every dashboard pageview / Friday turn / themed-brief render fans out
# 50 parallel Yahoo + Twelve Data calls. Single page hit = 50 Yahoo
# requests; 3-page navigation = 150. With 5-min cache the first hit
# warms it, the next 5 minutes of traffic all share the result.
_SNAPSHOT_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_SNAPSHOT_TTL_SEC = 5 * 60
_SNAPSHOT_LOCK = asyncio.Lock()


async def get_snapshot() -> dict[str, Any]:
    """Return a compact market snapshot suitable for brief + Friday context.

    5-min TTL cache (audit-round 2026-04-30) — concurrent callers
    serialize on the lock so a cache miss doesn't fan out to multiple
    parallel ~50-ticker Yahoo bursts.

    Empty `rows` and empty `one_liner` on total failure (fail-soft).
    """
    import time as _time
    now = _time.time()
    if _SNAPSHOT_CACHE["payload"] and (now - _SNAPSHOT_CACHE["at"]) < _SNAPSHOT_TTL_SEC:
        return _SNAPSHOT_CACHE["payload"]
    async with _SNAPSHOT_LOCK:
        # Re-check after acquiring lock — another coroutine may have
        # filled the cache while we waited.
        now = _time.time()
        if _SNAPSHOT_CACHE["payload"] and (now - _SNAPSHOT_CACHE["at"]) < _SNAPSHOT_TTL_SEC:
            return _SNAPSHOT_CACHE["payload"]
        async with httpx.AsyncClient(follow_redirects=True) as client:
            results = await asyncio.gather(
                *[_fetch_one(client, t) for t in TICKERS],
                return_exceptions=False,
            )
        rows = [r for r in results if r]
        payload = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
            "one_liner": _build_one_liner(rows),
            "groups": grouped_for_dashboard(rows),
            "aed_usd_peg": 3.6725,
        }
        _SNAPSHOT_CACHE["at"] = now
        _SNAPSHOT_CACHE["payload"] = payload
        return payload


def format_for_brief(snapshot: dict[str, Any]) -> str:
    """Single string for injection into the brief writer's task context."""
    line = snapshot.get("one_liner") or ""
    if not line:
        return "Market snapshot: unavailable."
    return line
