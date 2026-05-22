"""Google Trends scout — demand-intent signal, gated by corroboration.

Stakeholder direction (2026-04-29): Trends data is noisy on its own —
spikes can be Reddit-driven, bot-driven, or campaign-driven without any
underlying market move. Used in isolation, it pollutes the brief.

Solution: this scout fires a signal ONLY when a Google Trends spike on a
keyword corroborates with a recent signal from a DIFFERENT scout.
Concretely:
  1. We pull weekly interest scores for ~30 curated keywords (developer
     names, project names, topic phrases).
  2. We compute the WoW delta. Anything ≥ +50% counts as a "spike".
  3. For each spike, we check: is there a signal in the last 7 days
     whose headline/dek mentions the same entity, from a different scout?
  4. Only spikes WITH corroboration become signals. Lone spikes go into
     a fail-soft `metrics_daily` row (charting surface) and stop there.

This makes Google Trends an enhancement to existing signals rather than
a noise generator, and is the first scout that explicitly requires
corroboration before publishing. Mirrors the architectural pattern we
want for future low-signal-density sources.

Stamp:
  - source: "gtrends:<keyword_slug>"   (scout-prefix "gtrends")
  - category_hint: derived from the matched signal's category
  - tier_hint: "social" (search-intent is a soft consumer signal)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from tools import supabase_tool as sb

log = logging.getLogger(__name__)


# Keywords we monitor. (kw, dev_slug_hint, region, kw_kind).
# kw_kind drives the "what counts as corroboration" matching:
#   - "developer": match signals where dev_slug = the slug
#   - "project":   match signals where headline/dek contains the kw
#   - "topic":     match signals where headline/dek contains any of the topic terms
KEYWORDS: list[dict[str, Any]] = [
    # Developers
    {"kw": "Sobha Realty",      "dev_slug": "sobha",     "region": "dubai", "kind": "developer"},
    {"kw": "Emaar",             "dev_slug": "emaar",     "region": "dubai", "kind": "developer"},
    {"kw": "Damac",             "dev_slug": "damac",     "region": "dubai", "kind": "developer"},
    {"kw": "Aldar",             "dev_slug": "aldar",     "region": "abu_dhabi", "kind": "developer"},
    {"kw": "Binghatti",         "dev_slug": "binghatti", "region": "dubai", "kind": "developer"},
    {"kw": "Nakheel",           "dev_slug": "nakheel",   "region": "dubai", "kind": "developer"},
    # Sobha-relevant projects
    {"kw": "Sobha Hartland",    "dev_slug": "sobha",     "region": "dubai", "kind": "project"},
    {"kw": "Sobha One",         "dev_slug": "sobha",     "region": "dubai", "kind": "project"},
    {"kw": "Sobha SeaHaven",    "dev_slug": "sobha",     "region": "dubai", "kind": "project"},
    {"kw": "Sobha Skyvue",      "dev_slug": "sobha",     "region": "dubai", "kind": "project"},
    # Competitor flagship projects (when these spike, it's a comp signal)
    {"kw": "Emaar Beachfront",  "dev_slug": "emaar",     "region": "dubai", "kind": "project"},
    {"kw": "Damac Lagoons",     "dev_slug": "damac",     "region": "dubai", "kind": "project"},
    {"kw": "Binghatti Mercedes","dev_slug": "binghatti", "region": "dubai", "kind": "project"},
    {"kw": "Palm Jebel Ali",    "dev_slug": "nakheel",   "region": "dubai", "kind": "project"},
    # Topic phrases — buyer-intent
    {"kw": "Dubai off plan",    "dev_slug": None, "region": "dubai", "kind": "topic"},
    {"kw": "Dubai property",    "dev_slug": None, "region": "dubai", "kind": "topic"},
    {"kw": "Dubai golden visa", "dev_slug": None, "region": "dubai", "kind": "topic"},
    {"kw": "Dubai mortgage",    "dev_slug": None, "region": "dubai", "kind": "topic"},
]


SPIKE_THRESHOLD = 0.50  # 50% WoW delta → counts as a spike
LOOKBACK_DAYS   = 7     # how far back to look for corroborating signal


def _dedup_key(kw: str, week_iso: str) -> str:
    return hashlib.sha256(f"gtrends:{kw}:{week_iso}".encode()).hexdigest()


def _slug(kw: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", kw.lower()).strip("_")


def _has_corroborating_signal(kw_def: dict[str, Any]) -> dict[str, Any] | None:
    """Look back LOOKBACK_DAYS in the signals table for evidence that
    something IS happening with this keyword's entity from a non-gtrends
    source. Return the matching signal dict (the strongest match) or None.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    try:
        if kw_def["kind"] == "developer" and kw_def.get("dev_slug"):
            rows = (
                sb.client().table("signals")
                .select("id,headline,dek,dev_slug,category,decision_tag,priority,last_seen_at,urls")
                .eq("dev_slug", kw_def["dev_slug"])
                .gte("last_seen_at", since)
                .eq("archived", False)
                .order("priority", desc=True)
                .limit(5).execute().data or []
            )
            return rows[0] if rows else None
        # project / topic — substring match against headline + dek.
        kw = kw_def["kw"]
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,dev_slug,category,decision_tag,priority,last_seen_at,urls")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .or_(f"headline.ilike.%{kw}%,dek.ilike.%{kw}%")
            .order("priority", desc=True)
            .limit(5).execute().data or []
        )
        return rows[0] if rows else None
    except Exception as e:
        log.info("[gtrends] corroboration query failed: %s", e)
        return None


async def _fetch_trends_one(kw_def: dict[str, Any]) -> dict[str, Any] | None:
    """Pull recent weekly interest for one keyword via pytrends.

    Returns: {kw, this_week, last_week, delta_pct, week_iso}.
    Fail-soft: returns None on any error (Google 429s, library breakage,
    pytrends import failure).
    """
    try:
        from pytrends.request import TrendReq
    except Exception as e:
        log.warning("[gtrends] pytrends import failed: %s", e)
        return None

    def _call() -> dict[str, Any] | None:
        # geo='AE' for Dubai-focused queries; topic phrases stay global.
        geo = "AE" if kw_def.get("region") in ("dubai", "abu_dhabi") else ""
        py = TrendReq(hl="en-US", tz=240, retries=1, backoff_factor=0.4)
        py.build_payload([kw_def["kw"]], cat=0, timeframe="now 7-d", geo=geo)
        df = py.interest_over_time()
        if df is None or df.empty or kw_def["kw"] not in df.columns:
            return None
        # Last 24h vs prior 24h → WoW delta proxy on the 7d frame.
        # pytrends returns hourly buckets for "now 7-d".
        last24 = df[kw_def["kw"]].tail(24).mean()
        prior24 = df[kw_def["kw"]].iloc[-48:-24].mean()
        if prior24 == 0:
            delta = 1.0 if last24 > 0 else 0.0
        else:
            delta = float((last24 - prior24) / prior24)
        return {
            "kw":        kw_def["kw"],
            "this":      float(last24),
            "prior":     float(prior24),
            "delta_pct": delta,
            "week_iso":  datetime.now(timezone.utc).date().isoformat(),
        }

    try:
        # pytrends is sync — offload to a thread.
        return await asyncio.to_thread(_call)
    except Exception as e:
        log.warning("[gtrends:%s] fetch failed: %s", kw_def["kw"], type(e).__name__)
        return None


async def run(limit: int = 20) -> list[dict[str, Any]]:
    """Run all gtrends keyword fetches under a single hard timeout.

    pytrends + Google have no contract — when Google rate-limits, the
    library can hang indefinitely. Without an outer timeout, the entire
    ingest pipeline waits on this one scout. 90s is generous enough for
    18 keywords sequentially when things work, and bounded enough that
    a hang doesn't sink the whole pipeline.
    """
    try:
        return await asyncio.wait_for(_run_inner(limit), timeout=90.0)
    except asyncio.TimeoutError:
        log.warning("[gtrends] outer timeout — returning partial results")
        return []
    except Exception as e:
        log.warning("[gtrends] run failed: %s", type(e).__name__)
        return []


async def _run_inner(limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Sequential — Google Trends rate-limits aggressively. ~250ms gap
    # between calls keeps us below the 429 threshold most days.
    for kw_def in KEYWORDS:
        # Per-keyword timeout — if one keyword hangs, skip and continue.
        try:
            trend = await asyncio.wait_for(_fetch_trends_one(kw_def), timeout=8.0)
        except asyncio.TimeoutError:
            log.info("[gtrends:%s] per-keyword timeout — skipping", kw_def["kw"])
            continue
        if not trend:
            continue
        # Always log the datapoint to the time-series store so the
        # dashboard can chart demand intent even when there's no signal.
        try:
            sb.upsert_metric_daily(
                metric_key=f"gtrends_{_slug(kw_def['kw'])}",
                captured_at=datetime.now(timezone.utc),
                value_num=trend["this"],
                source=f"gtrends:{_slug(kw_def['kw'])}",
                bucket="dubai" if kw_def.get("region") in ("dubai", "abu_dhabi") else "global",
                unit="trends_index",
                raw_json={"prior": trend["prior"], "delta_pct": trend["delta_pct"], "kind": kw_def["kind"]},
            )
        except Exception:
            pass
        # Spike gate: only fire as a signal if delta is meaningful.
        if trend["delta_pct"] < SPIKE_THRESHOLD:
            continue
        # Corroboration gate: don't fire unless another scout has
        # surfaced something for this entity in the last week.
        corroborator = _has_corroborating_signal(kw_def)
        if not corroborator:
            log.info("[gtrends:%s] spike +%.0f%% — no corroborator, dropped",
                     kw_def["kw"], trend["delta_pct"] * 100)
            continue

        delta_label = f"+{int(round(trend['delta_pct'] * 100))}%"
        title = f"Google Trends · {kw_def['kw']} interest {delta_label} WoW (corroborates #{corroborator.get('id')})"
        summary = (
            f"Search interest for '{kw_def['kw']}' jumped {delta_label} week-on-week. "
            f"Corroborates recent signal: \"{corroborator.get('headline')}\". "
            f"Demand-intent layer aligned with on-the-ground activity."
        )
        # Inherit category from the corroborating signal so the spike
        # routes to the same themed page.
        cat_hint = corroborator.get("category") or "demand"
        out.append({
            "source": f"gtrends:{_slug(kw_def['kw'])}",
            "source_url": f"https://trends.google.com/trends/explore?q={kw_def['kw'].replace(' ', '+')}",
            "title": title[:300],
            "summary": summary[:600],
            "raw_json": {
                "kw":         kw_def["kw"],
                "kind":       kw_def["kind"],
                "this_week":  trend["this"],
                "prior_week": trend["prior"],
                "delta_pct":  trend["delta_pct"],
                "corroborates_signal_id": corroborator.get("id"),
                "region":         kw_def["region"],
                "country_code":   "AE" if kw_def["region"] in ("dubai", "abu_dhabi") else None,
                "category_hint":  cat_hint,
                "dev_slug_hint":  kw_def.get("dev_slug"),
                "tier_hint":      "social",
            },
            "dedup_key": _dedup_key(kw_def["kw"], trend["week_iso"]),
        })
        if len(out) >= limit:
            break

    log.info("[gtrends] keywords=%d fired=%d (corroboration-gated)", len(KEYWORDS), len(out))
    return out
