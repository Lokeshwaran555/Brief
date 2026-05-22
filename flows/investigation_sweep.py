"""Scheduled investigation sweep.

Ingest fires investigations for priority=high signals at the moment
they're promoted. That misses two cases:

  1. Signals that existed before the auto-fire feature deployed.
  2. Rare race conditions where the auto-fire coroutine crashed
     silently before hitting investigation_flow.

This sweep runs hourly and picks up any priority=high signal from the
last 72 hours that does NOT have a market_event referencing it, then
fires an investigation. Hard cap of 5 per sweep tick so one stuck
investigation can't starve the rest.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from flows import investigation_flow
from tools import supabase_tool as sb
from tools.observability import observe

log = logging.getLogger(__name__)

MAX_PER_SWEEP = 5
LOOKBACK_HOURS = 72


def _find_uncovered_high_signals() -> list[dict[str, Any]]:
    """priority=high signals from the last 72h that have no market_event."""
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    client = sb.client()

    sig_rows = (
        client.table("signals")
        .select("id,headline,priority,last_seen_at,archived")
        .eq("priority", "high")
        .eq("archived", False)
        .gte("last_seen_at", since)
        .order("last_seen_at", desc=True)
        .limit(50)
        .execute()
        .data
        or []
    )
    if not sig_rows:
        return []

    sig_ids = [r["id"] for r in sig_rows]
    ev_rows = (
        client.table("market_events")
        .select("seed_signal_id")
        .in_("seed_signal_id", sig_ids)
        .execute()
        .data
        or []
    )
    covered = {r.get("seed_signal_id") for r in ev_rows if r.get("seed_signal_id")}
    return [r for r in sig_rows if r["id"] not in covered]


def _mark_stale_events() -> int:
    """Events not refreshed in 7d move to status='stale' so the dashboard
    can flag them. Investigations on those events bring them back to
    'ready'."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    try:
        # Only flip from 'ready' → 'stale'; leave 'conflicted' alone.
        rows = (
            sb.client()
            .table("market_events")
            .update({"status": "stale"})
            .eq("status", "ready")
            .lt("last_updated_at", cutoff)
            .execute()
            .data
            or []
        )
        return len(rows)
    except Exception as e:
        log.warning("mark_stale_events failed: %s", e)
        return 0


@observe(name="investigation_sweep")
async def run() -> dict[str, Any]:
    """Mark stale events + fire investigations for uncovered priority=high signals."""
    started = datetime.now(timezone.utc)
    stale_marked = _mark_stale_events()
    uncovered = _find_uncovered_high_signals()
    log.info(
        "investigation_sweep: %d uncovered, %d marked stale",
        len(uncovered),
        stale_marked,
    )

    # 2026-04-30 audit-round: respect the same _INVESTIGATION_SEM the
    # ingest flow uses (cap=2 concurrent) so the hourly sweep can't
    # blow past the documented "≤2 concurrent" cap when fired during
    # an in-progress ingest. Lazy-import to avoid circular imports.
    try:
        from flows.ingest_flow import _INVESTIGATION_SEM as _SEM
    except Exception:
        _SEM = None
    fired: list[dict[str, Any]] = []
    for sig in uncovered[:MAX_PER_SWEEP]:
        try:
            if _SEM is not None:
                async with _SEM:
                    result = await investigation_flow.run(int(sig["id"]), trigger="sweep")
            else:
                result = await investigation_flow.run(int(sig["id"]), trigger="sweep")
            fired.append(
                {
                    "signal_id": sig["id"],
                    "headline": sig.get("headline"),
                    "ok": bool(result.get("ok")),
                    "event_id": result.get("event_id"),
                }
            )
        except Exception as e:
            log.warning("sweep: investigation failed for %s: %s", sig.get("id"), e)
            fired.append({"signal_id": sig["id"], "ok": False, "error": str(e)[:200]})

    finished = datetime.now(timezone.utc)
    return {
        "uncovered_total": len(uncovered),
        "stale_marked": stale_marked,
        "fired": fired,
        "duration_s": (finished - started).total_seconds(),
    }
