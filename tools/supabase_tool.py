"""Thin Supabase wrapper. Service-role key — bypasses RLS.

Week-1 operations only: upsert intel_raw, insert intel_scored,
upsert signals with title-hash cluster_key.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from settings import settings

_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _client


def title_hash(title: str) -> str:
    """Normalised-title sha256 — fallback dedup when embedding is unavailable."""
    norm = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def find_similar_signal(
    embedding: list[float],
    threshold: float | None = None,
    window_days: int | None = None,
) -> dict[str, Any] | None:
    """Return the closest existing signal by cosine similarity, or None.

    Uses the `match_signal` RPC defined in db/migrations/002_pgvector_dedup.sql.
    Returns None on any RPC error so ingest can degrade to title-hash.
    """
    if not embedding:
        return None
    try:
        params = {
            "query_embedding": embedding,
            "match_threshold": threshold if threshold is not None else settings.signal_dedup_similarity,
            "match_count": 1,
            "window_days": window_days if window_days is not None else settings.signal_dedup_window_days,
        }
        rows = client().rpc("match_signal", params).execute().data or []
        return rows[0] if rows else None
    except Exception as e:
        # Most likely cause: migration not yet applied. Log once-per-run, fall through.
        import logging
        logging.getLogger(__name__).warning("match_signal RPC failed: %s", e)
        return None


def upsert_metric_daily(
    metric_key: str,
    captured_at: datetime,
    *,
    value_num: float | int | None = None,
    value_text: str | None = None,
    source: str,
    bucket: str | None = None,
    unit: str | None = None,
    raw_json: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Write a single (metric_key, captured_at) datapoint to the
    `metrics_daily` time-series store. Upserts on the unique index so
    re-running a scout on the same trading day overwrites cleanly.

    Used by:
      - tools/market_snapshot.py — writes ticker closes daily
      - eibor_scout — fixings
      - materials_scout — BDI/SCFI/cement/rebar
      - cme_fedwatch_scout — implied prob of next Fed move
      - sukuk_scout — bond yields when extractable

    Fail-soft: returns None on any error so callers never block ingest.
    """
    if not metric_key or not source:
        return None
    payload = {
        "metric_key":  metric_key,
        "captured_at": captured_at.isoformat(),
        "value_num":   None if value_num is None else float(value_num),
        "value_text":  value_text,
        "source":      source,
        "bucket":      bucket,
        "unit":        unit,
        "raw_json":    raw_json or {},
    }
    try:
        return (
            client().table("metrics_daily")
            .upsert(payload, on_conflict="metric_key,captured_at")
            .execute().data[0]
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).info(
            "upsert_metric_daily(%s): %s — run migration 017_metrics_daily.sql",
            metric_key, type(e).__name__,
        )
        return None


def insert_intel_raw(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upsert by dedup_key so re-runs don't create duplicate raw rows.

    Two scouts can produce the same dedup_key in a single batch (e.g.
    when both pick up the same press release from different domains
    that hash to the same headline+host). Postgres rejects an
    ON CONFLICT DO UPDATE that touches the same row twice in one
    command with code 21000, so we de-dupe within the batch first —
    last write wins, which matches the per-scout independence we want.
    """
    if not rows:
        return []
    seen: dict[str, dict[str, Any]] = {}
    leftover: list[dict[str, Any]] = []
    for r in rows:
        k = r.get("dedup_key")
        if k:
            seen[k] = r  # last wins
        else:
            leftover.append(r)
    deduped = list(seen.values()) + leftover
    return (
        client()
        .table("intel_raw")
        .upsert(deduped, on_conflict="dedup_key")
        .execute()
        .data
    )


def insert_intel_scored(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return client().table("intel_scored").insert(row).execute().data[0]
    except Exception as e:
        # 2026-04-30 round 7: fail-soft if migration 019 (subcategory)
        # hasn't run on intel_scored yet — drop the field and retry.
        msg = str(e).lower()
        if "subcategory" in msg:
            import logging
            logging.getLogger(__name__).warning(
                "intel_scored.insert: subcategory column missing — "
                "run migration 019_subcategory.sql. Retrying without."
            )
            row.pop("subcategory", None)
            return client().table("intel_scored").insert(row).execute().data[0]
        raise


def _source_prefix(source: str | None) -> str | None:
    """Extract the scout-family prefix from a raw signal's `source` field.

    Source strings look like: 'instagram:stake_uae', 'proptech:ai_design',
    'gnews:dubai-launches', 'linkedin:themed-tech'. The prefix before the
    colon is the SCOUT identity — different scouts = independent sources
    for corroboration scoring (2026-04-29). Same scout firing twice on
    different sub-targets is NOT independent corroboration.
    """
    if not source:
        return None
    s = str(source).strip()
    if not s:
        return None
    return s.split(":", 1)[0]


def upsert_signal(
    row: dict[str, Any],
    embedding: list[float] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Cluster by semantic similarity (pgvector) first, then exact cluster_key.

    Resolution order:
      1. If `embedding` is provided and a similar recent signal exists
         (cosine ≥ settings.signal_dedup_similarity), merge into it.
      2. Else if a row with the same `cluster_key` (title-hash) exists, merge.
      3. Else insert a new signal (with embedding if provided).

    Cross-source corroboration scoring (2026-04-29):
      - `source` arg carries the raw signal's scout source (e.g.
        'instagram:stake_uae'). At merge time, we increment
        `corroboration_count` ONLY when the new source's family-prefix
        isn't already in `corroborating_sources` — so independent scouts
        confirming the same event raise the score, but the same scout
        firing twice doesn't.
      - On fresh insert, seed corroborating_sources with the new prefix.
    """
    sb = client()
    now = datetime.now(timezone.utc).isoformat()
    src_prefix = _source_prefix(source)

    prev: dict[str, Any] | None = None
    if embedding:
        match = find_similar_signal(embedding)
        if match:
            existing = (
                sb.table("signals").select("*").eq("id", match["id"]).execute().data
            )
            if existing:
                prev = existing[0]
    if prev is None:
        existing = (
            sb.table("signals").select("*").eq("cluster_key", row["cluster_key"]).execute().data
        )
        if existing:
            prev = existing[0]

    if prev:
        merged: dict[str, Any] = {
            "last_seen_at": now,
            "source_count": prev["source_count"] + 1,
            "urls": list({*(prev.get("urls") or []), *(row.get("urls") or [])}),
            "priority": _max_priority(prev.get("priority"), row.get("priority")),
            "confidence": max(
                float(prev.get("confidence") or 0), float(row.get("confidence") or 0)
            ),
        }
        # Corroboration: only count if this scout-family prefix is new.
        prior_sources = prev.get("corroborating_sources") or []
        if isinstance(prior_sources, str):
            try:
                import json as _json
                prior_sources = _json.loads(prior_sources)
            except Exception:
                prior_sources = []
        if src_prefix and src_prefix not in prior_sources:
            merged["corroborating_sources"] = list(prior_sources) + [src_prefix]
            merged["corroboration_count"] = int(prev.get("corroboration_count") or 1) + 1
        # Backfill embedding if the existing row was created pre-pgvector.
        if embedding and not prev.get("embedding"):
            merged["embedding"] = embedding
        return (
            sb.table("signals")
            .update(merged)
            .eq("id", prev["id"])
            .execute()
            .data[0]
        )

    row.setdefault("first_seen_at", now)
    row.setdefault("last_seen_at", now)
    # Seed corroboration on insert. corroboration_count defaults to 1 in
    # the schema; we set the source list to [prefix] so future merges
    # correctly detect "this prefix already counted".
    if src_prefix:
        row.setdefault("corroborating_sources", [src_prefix])
        row.setdefault("corroboration_count", 1)
    if embedding:
        row["embedding"] = embedding
    try:
        return sb.table("signals").insert(row).execute().data[0]
    except Exception as e:
        # Migration 008_signal_region.sql adds `region` + `country_code`
        # to signals. If it hasn't run yet, the insert errors. Retry
        # without those fields so ingest doesn't break the whole run
        # — region tagging just won't persist until migration applies.
        # Migration 016_signal_quality.sql adds corroboration columns
        # — same pattern: drop and retry if missing.
        msg = str(e).lower()
        if "region" in msg or "country_code" in msg:
            import logging
            logging.getLogger(__name__).warning(
                "signals.insert: region/country columns missing — "
                "run migration 008_signal_region.sql. Retrying without."
            )
            row.pop("region", None)
            row.pop("country_code", None)
            return sb.table("signals").insert(row).execute().data[0]
        if "corroborat" in msg:
            import logging
            logging.getLogger(__name__).warning(
                "signals.insert: corroboration columns missing — "
                "run migration 016_signal_quality.sql. Retrying without."
            )
            row.pop("corroborating_sources", None)
            row.pop("corroboration_count", None)
            return sb.table("signals").insert(row).execute().data[0]
        if "subcategory" in msg:
            import logging
            logging.getLogger(__name__).warning(
                "signals.insert: subcategory column missing — "
                "run migration 019_subcategory.sql. Retrying without."
            )
            row.pop("subcategory", None)
            return sb.table("signals").insert(row).execute().data[0]
        raise


def mark_raw_processed(raw_ids: list[int]) -> None:
    if not raw_ids:
        return
    client().table("intel_raw").update({"processed": True}).in_("id", raw_ids).execute()


_PRI_RANK = {"high": 3, "medium": 2, "low": 1}


def _max_priority(a: str | None, b: str | None) -> str | None:
    ra, rb = _PRI_RANK.get((a or "").lower(), 0), _PRI_RANK.get((b or "").lower(), 0)
    return a if ra >= rb else b
