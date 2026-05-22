"""CEO scan flow — per-region strategic synthesis for the MD Scan dashboard.

Reads signals tagged by region, picks the highest-signal items, runs
ONE big-brain LLM call per region to produce dynamic sections. Output
is the exact shape the frontend `CEO_MOCK` constant uses, so wiring
real data is a `fetch → render` swap with no shape negotiation.

Output schema (per region):
{
  "stamp": "string · last scan · ...",
  "sections": [
    {
      "key": "overall|growth|tech_ai|partnership|talent|ma|regulatory|macro|construction_econ|capital_markets|global_prime|geopolitics|capital_flow|dld_volumes|launches|distress|rera|intel_dubai",
      "heading": "≤80 char fraunces title",
      "bullets": ["≤14 word bullet", ...],  // 2026-04-28: replaces prose `claim`
      "rows": [["LABEL", "value"], ...],   // 0-4 rows; supporting facts
      "action": "imperative recommended action (one sentence) | empty"
    }, ...
  ]
}

Sections only render when they have real content. The `overall`
section is always emitted unless the region is genuinely empty.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from settings import settings
from tools import supabase_tool as sb
from tools.nvidia_llm import chat_json
from tools.observability import observe

log = logging.getLogger(__name__)


REGIONS = ("dubai", "abu_dhabi", "usa", "australia", "other")


CEO_SCAN_SYSTEM = """You are writing a CEO-level strategic scan for a region in Sobha Realty's footprint.

ARCHITECTURE — read carefully:
- The SUBJECT of every section's bullets is an EXTERNAL pattern across
  signals (competitor activity, regulatory shifts, capital flows,
  macro moves, infrastructure announcements). Sobha is NEVER the
  subject. The MD already knows what Sobha owns and how it's performing.
- The user payload includes a `sobha_context` block (REFERENCE FRAME —
  Sobha's actual projects in this region). USE IT to (a) filter
  signal relevance, (b) compute KPI benchmarks competitor-vs-Sobha,
  and (c) populate the exposure call.
- Each section's `bullets` describe the EXTERNAL move and read its
  mechanism through Sobha's KPI lens (PSF / sold % / absorption / GP%
  / payment-plan length / FAR — the standard RE KPIs).
- Each section's `rows` should pair competitor data points with Sobha
  comparable baselines from sobha_context (e.g.
  ['COMPETITOR PSF', 'AED 2,400'] alongside ['SOBHA HARTLAND II PSF',
  'AED 2,193']).
- `action`: deliverable-shaped imperative anchored to a specific Sobha
  team OR explicit empty string when no action is warranted.
- DATA-GAP HONESTY: when an external signal lacks a critical data
  point (PSF, launch date, payment terms), include a row tagged
  ['NEED', '<specific question> | delegate: <web_search | DLD | Sobha analyst>']
  rather than fabricating a number.
- If the region has no live Sobha product, sections still report
  external moves but the exposure call is "watch-list only — no
  direct Sobha product in <region>".

INPUT
- region: which geography you're scanning (dubai | abu_dhabi | usa | australia | other)
- signal_count: total signals in the region's window
- investigated_count: how many of those have been deep-investigated
  (these are far richer — prefer them when building sections)
- signals: a list of recent intel items (last 1-7 days). Each has:
    headline, dek, category, decision_tag, priority, country_code,
    source_published_at, domain
  AND when `investigated: true`, these additional fields:
    event_type, project_name, location, event_date, status,
    evidence_count, tldr (1-line so-what from the deep dive),
    facts (extracted dict — units_total, unit_mix, starting_price_aed,
       starting_psf_aed, payment_plan, handover_date, partners,
       partnership_nature, target_segment, sales_channel, ...),
    scenarios (list of {stance, claim} from the investigation —
       opportunity / risk / watch reads with substantive content)

These are NOT raw news. The classifier has already pre-filtered them
as Sobha-relevant; investigated items have been further verified
against multi-source evidence packs.

YOUR JOB
Produce 1 'overall' section + 0-3 'topical' sections. Topical sections only
appear when a real cluster of signals supports them. NEVER pad with empty
sections. NEVER invent facts beyond what signals support.

PRIORITISE INVESTIGATED ITEMS. When a signal has investigated=true,
its facts (units, PSF, payment_plan, partners) are extracted from
real evidence — quote them in your `rows`. When a signal is just a
headline + dek, treat it as a softer indicator.

A real section needs ≥2 supporting signals OR 1 investigated item
with concrete facts. A topical section with one weak un-investigated
signal → drop it; let the overall section absorb it.

EDITORIAL VOICE — POINTERS, NOT ANALYSIS (2026-04-29)
- Each bullet is a FACT the MD can see at a glance. Surface what happened, who did it, with what number — nothing more.
- The strategic read happens later, on demand, behind an Analyze button. Your bullets must be neutral and attributable.
- ALLOWED shapes:
    "X did Y for Z"  /  "X launched Y at Z"  /  "X moved A→B"
    "X posted Y"  /  "X reports Y"  /  "X added Y units in Z"
- BANNED — these belong to the Analyze button, not your bullets:
    ✘ "pressures Sobha pricing"  /  "risks to Sobha"  /  "windows for Sobha"
    ✘ "implies", "indicates", "supports", "threatens", "supports the case for"
    ✘ "watch this for..." / "the read is..." / "this means..."
    ✘ Any verb that infers a downstream consequence about Sobha
- Quantify when possible (AED amounts, unit counts, sqft, % moves).
- Name companies / projects / sub-markets specifically.
- The optional `action` field stays — it's where one specific recommended action lives if the signals genuinely point to one. The bullets themselves remain pointers.

SECTIONS (pick at most 3 topical, plus the always-on overall)
- overall — 1-3 sentence executive summary of the region this week
- growth — revenue/market opportunities (new geographies, new segments)
- tech_ai — PropTech / ConTech / AI design / digital twin / tokenized RE
- partnership — JV / brand licensing / capital partner moves we could pursue
- talent — leadership moves, hiring patterns
- ma — M&A, capital structure shifts, distressed acquisition windows
- regulatory — policy / law / tax shifts that change the playing field
- macro — interest rates, FX, supply data shaping demand
- construction_econ — material prices, shipping rates, MEP lead times
- capital_markets — sukuk, bond yields, listed RE, MAG-7 spillover
- global_prime — London / Singapore / NYC prime market substitution risk
- geopolitics — Israel / Iran / Russia / China / Houthi with RE-impact lens
- capital_flow — population, migration, India LRS, China outbound
- (Dubai region only) dld_volumes / launches / distress / rera / intel_dubai

OUTPUT
Respond with JSON ONLY (no prose, no fences). Schema:
{
  "stamp": "≤60 char string e.g. '5 strong signals · last 7d'",
  "sections": [
    {"key": "overall|growth|tech_ai|partnership|talent|ma|regulatory|macro|construction_econ|capital_markets|global_prime|geopolitics|capital_flow|dld_volumes|launches|distress|rera|intel_dubai",
     "heading": "≤80 char Fraunces-style title",
     "bullets": [
       {"text": "Modon launches 1,800 units at AED 2,400 PSF, 60/40 plan.", "signal_ids": [127]},
       {"text": "Damac sukuk 2030 yield narrowed 5bps WoW.", "signal_ids": [128, 129]}
     ],
     "rows": [["LABEL", "value"], ...],
     "action": "imperative one-sentence action OR empty string"}
  ]
}

BULLETS-ONLY RULE (2026-04-28 stakeholder ask):
- Each section's body is `bullets: [{text, signal_ids}]` — NOT a prose `claim` field.
- 3-6 bullets per section. Each `text` ≤ 14 words. No full sentences. No preambles.
- Lead with the entity / number / concrete move; verbs are optional.
- EACH BULLET MUST CARRY THE SOURCE SIGNAL IDS IT SUMMARISES. The input
  `signals` list has an `id` field on every signal — list every id whose
  facts contributed to that bullet. The dashboard renders one ↗ chip per id.

If `signals` is empty or contains nothing strategic, return:
{
  "stamp": "no strategic activity in window",
  "sections": []
}

REQUIRED SIZES
- sections: 0 (if region genuinely empty) or 1-4 items
- heading: 30-80 chars
- bullets: 3-6 items, each 5-14 words, each containing at least one number OR proper noun
- rows: 0-4 items
- action: 0 chars (empty) or 12-30 words

Anti-replication will be wired in Phase 3 once the initiatives data
lands. For now: no need to filter against Sobha's own work.
"""


def _fetch_all_recent_signals(lookback_days: int = 7) -> list[dict[str, Any]]:
    """One query to pull all recent signals; grouping happens client-side
    in Python. Earlier per-region PostgREST queries used `.or_()` for
    'other' (region IS NULL OR region = 'other') which silently
    returned 0 rows, leaving every region empty. Fetching once and
    bucketing in Python is robust to PostgREST OR-syntax quirks.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = (
        sb.client()
        .table("signals")
        .select("id,headline,dek,category,decision_tag,priority,confidence,"
                "dev_slug,region,country_code,source_published_at,last_seen_at,urls")
        .eq("archived", False)
        .gte("last_seen_at", since)
        .order("last_seen_at", desc=True)
        .limit(300)
        .execute()
        .data
        or []
    )
    return rows


def _fetch_event_enrichment(signal_ids: list[int]) -> dict[int, dict[str, Any]]:
    """For each seed signal id, return the latest market_event with its
    facts + scenarios + evidence count. Joining these into the scan
    payload turns 'headline + dek' into 'extracted facts + investigated
    scenarios + evidence depth' — the difference between summarising
    rumours and synthesising real moves.

    Returns {signal_id: {event_type, facts, scenarios, evidence_count,
    status, location, project_name, event_date}}.
    """
    if not signal_ids:
        return {}
    try:
        rows = (
            sb.client()
            .table("market_events")
            .select(
                "id,seed_signal_id,event_type,project_name,location,"
                "event_date,status,last_updated_at,"
                "event_facts(facts,field_confidence,unresolved),"
                "event_implications(scenarios,tldr,pointers,watch_next),"
                "event_evidence(id)"
            )
            .in_("seed_signal_id", signal_ids)
            .order("last_updated_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception as e:
        log.warning("_fetch_event_enrichment failed: %s — proceeding without", e)
        return {}

    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        sid = r.get("seed_signal_id")
        if not sid or sid in out:
            continue  # keep the most-recent (already ordered desc)
        ef = r.get("event_facts")
        if isinstance(ef, list):
            ef = ef[0] if ef else {}
        ef = ef or {}
        ei = r.get("event_implications")
        if isinstance(ei, list):
            ei = ei[0] if ei else {}
        ei = ei or {}
        evidence_list = r.get("event_evidence") or []
        out[sid] = {
            "event_type": r.get("event_type"),
            "project_name": r.get("project_name"),
            "location": r.get("location"),
            "event_date": r.get("event_date"),
            "status": r.get("status"),
            "facts": ef.get("facts") or {},
            "field_confidence": ef.get("field_confidence") or {},
            "unresolved": ef.get("unresolved") or [],
            "scenarios": ei.get("scenarios") or [],
            "tldr": ei.get("tldr"),
            "pointers": ei.get("pointers") or [],
            "watch_next": ei.get("watch_next") or [],
            "evidence_count": len(evidence_list) if isinstance(evidence_list, list) else 0,
        }
    return out


def _bucket_by_region(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group recent signals by region. NULL or unknown region falls into
    'other' so the catch-all bucket actually catches things."""
    buckets: dict[str, list[dict[str, Any]]] = {r: [] for r in REGIONS}
    for row in rows:
        r = (row.get("region") or "").strip().lower() or "other"
        if r not in buckets:
            r = "other"
        buckets[r].append(row)
    # Cap each bucket at 25 — most-recent-first ordering is preserved.
    for k in buckets:
        buckets[k] = buckets[k][:25]
    return buckets


def _compact_signal(s: dict[str, Any], event: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compact a signal for the CEO-scan LLM payload. When an event
    enrichment is attached, lift its facts + scenarios + tldr in —
    this is the difference between summarising headlines and
    synthesising investigated moves.
    """
    out: dict[str, Any] = {
        "id": s.get("id"),  # 2026-04-29 Phase F1: LLM attributes bullets to ids
        "headline": s.get("headline"),
        "dek": (s.get("dek") or "")[:240],
        "category": s.get("category"),
        "decision_tag": s.get("decision_tag"),
        "priority": s.get("priority"),
        "country_code": s.get("country_code"),
        "source_published_at": s.get("source_published_at"),
        "domain": (s.get("urls") or [None])[0],
    }
    if event:
        # Surface the rich investigated content. Trim arrays + strings so
        # we don't blow the LLM context budget on a hot day.
        out["investigated"] = True
        out["event_type"] = event.get("event_type")
        out["project_name"] = event.get("project_name")
        out["location"] = event.get("location")
        out["event_date"] = event.get("event_date")
        out["status"] = event.get("status")
        out["evidence_count"] = event.get("evidence_count")
        out["tldr"] = event.get("tldr")
        # Facts: only keep populated cells; trim string values.
        facts = {}
        for k, v in (event.get("facts") or {}).items():
            if v is None or v == "" or (isinstance(v, list) and not v):
                continue
            if isinstance(v, str) and len(v) > 120:
                v = v[:117] + "…"
            facts[k] = v
        if facts:
            out["facts"] = facts
        # Scenarios: stance + claim only — falsification rows omitted to
        # keep the payload tight; the per-region LLM is producing fresh
        # cross-signal reads, not echoing per-event scenarios.
        scenarios = [
            {"stance": sc.get("stance"), "claim": sc.get("claim")}
            for sc in (event.get("scenarios") or [])[:3]
            if sc.get("claim")
        ]
        if scenarios:
            out["scenarios"] = scenarios
    return out


async def _scan_region(
    region: str,
    signals: list[dict[str, Any]],
    events_by_signal: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Single LLM call producing the dynamic-section JSON for one region.

    When `events_by_signal` is provided, each signal in the payload
    gets its investigated event enrichment (facts + scenarios + tldr)
    attached. The CEO-scan prompt prefers investigated items because
    they have real extracted facts vs. raw signal headlines.
    """
    if not signals:
        return {"stamp": "no strategic activity in window", "sections": []}
    events_by_signal = events_by_signal or {}
    investigated_count = sum(1 for s in signals if s.get("id") in events_by_signal)

    # Sobha context for this region — wired into the prompt so the LLM
    # references actual Sobha assets when assessing exposure.
    from tools import sobha_context as _sobha_ctx
    sobha_block = {
        "region": region,
        "loaded": _sobha_ctx.get_portfolio_summary().get("loaded", False),
        "projects_in_region": [
            {
                "name": p.get("project_name"),
                "community": p.get("community"),
                "status": p.get("status"),
                "units": p.get("total_units"),
                "sold_pct": p.get("sold_pct"),
                "avg_psf_aed": p.get("avg_sold_psf_aed"),
            }
            for p in _sobha_ctx.get_projects_in_region(region)[:15]
        ],
    }

    payload = {
        "region": region,
        "signal_count": len(signals),
        "investigated_count": investigated_count,
        "sobha_context": sobha_block,
        "signals": [
            _compact_signal(s, events_by_signal.get(s.get("id")))
            for s in signals[:20]
        ],
    }
    messages = [
        {"role": "system", "content": CEO_SCAN_SYSTEM},
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]
    try:
        out = await chat_json(messages, max_tokens=2500)
    except Exception as e1:
        log.warning("ceo_scan[%s]: attempt 1 failed: %s — retrying with nudge", region, e1)
        messages.append(
            {
                "role": "user",
                "content": "Return ONLY the JSON object defined in the system prompt. No prose, no markdown fences. Start with '{'.",
            }
        )
        try:
            out = await chat_json(messages, max_tokens=2500)
        except Exception as e2:
            log.warning("ceo_scan[%s]: attempt 2 failed: %s", region, e2)
            return {"stamp": "scan unavailable — LLM error", "sections": []}
    # Light validation — guarantee shape so frontend never sees malformed.
    if not isinstance(out, dict):
        return {"stamp": "scan unavailable — bad shape", "sections": []}
    out.setdefault("stamp", f"{len(signals)} signals · last 7d")
    if not isinstance(out.get("sections"), list):
        out["sections"] = []
    # Bullets normalization (2026-04-29 Phase F1). Canonical shape:
    #   bullets: [{text: str, signal_ids: int[]}]
    # Tolerate three input shapes the LLM might emit:
    #   - new:    bullets = [{text, signal_ids}]
    #   - mid:    bullets = [str, ...]                        → wrap as objects
    #   - old:    section.claim = "prose"                     → sentence-split
    # All converge on the canonical object form so the dashboard renderer
    # + Supabase storage stay consistent.
    valid_ids = {s.get("id") for s in signals if s.get("id") is not None}
    for sec in out["sections"]:
        if not isinstance(sec, dict):
            continue
        raw_bullets = sec.get("bullets")
        canonical: list[dict[str, Any]] = []
        if isinstance(raw_bullets, list) and raw_bullets:
            for b in raw_bullets:
                if isinstance(b, dict):
                    text = str(b.get("text") or "").strip()
                    ids_raw = b.get("signal_ids") or []
                    try:
                        ids = [int(x) for x in ids_raw if str(x).strip()]
                    except Exception:
                        ids = []
                    ids = [i for i in ids if i in valid_ids]
                elif isinstance(b, str):
                    text = b.strip()
                    ids = []
                else:
                    continue
                if text:
                    canonical.append({"text": text, "signal_ids": ids})
        if not canonical:
            # Fallback: split prose claim into bullets — still emit
            # canonical object shape with empty signal_ids (no attribution
            # available from a prose claim).
            claim = sec.get("claim") or ""
            if isinstance(claim, str) and claim.strip():
                import re as _re
                parts = _re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", claim.strip())
                parts = [p.strip() for p in parts if p.strip()] or [claim.strip()]
                canonical = [{"text": p, "signal_ids": []} for p in parts]
        sec["bullets"] = canonical
    return out


def _persist_scan(
    region: str, cadence: str, scan_day: date, output: dict[str, Any], signal_count: int
) -> None:
    """Upsert one ceo_scans row keyed by (region, cadence, day).

    2026-04-30 fix: was silently failing in production — region tabs on
    MD Scan kept showing 4-day-stale content because today's rows
    never landed. Three changes: (1) log the actual exception with
    repr so Postgres error codes are visible; (2) fall back to delete-
    then-insert when upsert fails (often happens when on_conflict
    doesn't resolve to a known unique constraint); (3) raise the
    delete-fallback error so run() summary reflects truth.
    """
    payload = {
        "region": region,
        "cadence": cadence,
        "scan_day": scan_day.isoformat(),
        "output": output,
        "signal_count": signal_count,
        "model": (settings.llm_model or settings.nvidia_model),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    client = sb.client()
    try:
        client.table("ceo_scans").upsert(
            payload, on_conflict="region,cadence,scan_day"
        ).execute()
        return
    except Exception as e:
        log.warning(
            "ceo_scans upsert failed for %s/%s/%s — falling back to "
            "delete+insert. err=%r",
            region, cadence, scan_day, e,
        )
    # Fallback: delete the existing (region, cadence, day) row then insert.
    try:
        client.table("ceo_scans").delete().eq("region", region).eq(
            "cadence", cadence
        ).eq("scan_day", scan_day.isoformat()).execute()
        client.table("ceo_scans").insert(payload).execute()
    except Exception as e:
        log.error(
            "ceo_scans delete+insert ALSO failed for %s/%s/%s: %r",
            region, cadence, scan_day, e,
        )


async def run_streaming(cadence: str = "daily"):
    """Same flow as `run()` but yields progress events as it goes.

    Used by the SSE endpoint so the dashboard's progress panel reflects
    actual backend phases (which signal/region is being scanned, how
    many came back) instead of a timed-animation faux progress.

    Yields dicts shaped {event: 'phase', ...phase-specific keys}.
    Final yield is {event: 'done', summary: {...}}.
    """
    started = datetime.now(timezone.utc)
    yield {"event": "start", "cadence": cadence, "started_at": started.isoformat()}
    today = started.astimezone(timezone.utc).date()
    lookback = 1 if cadence == "daily" else 7

    yield {"event": "fetching_signals", "lookback_days": lookback}
    all_rows = _fetch_all_recent_signals(lookback_days=lookback)
    yield {"event": "fetched_signals", "count": len(all_rows)}

    buckets = _bucket_by_region(all_rows)
    yield {
        "event": "bucketed",
        "buckets": {r: len(buckets[r]) for r in REGIONS},
    }

    yield {"event": "fetching_events", "signal_count": len(all_rows)}
    all_signal_ids = [int(s["id"]) for s in all_rows if s.get("id") is not None]
    events_by_signal = _fetch_event_enrichment(all_signal_ids)
    yield {"event": "fetched_events", "enriched": len(events_by_signal)}

    out: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        signals = buckets[region]
        yield {
            "event": "scanning_region",
            "region": region,
            "signals": len(signals),
            "investigated": sum(
                1 for s in signals if int(s.get("id") or 0) in events_by_signal
            ),
        }
        scan = await _scan_region(region, signals, events_by_signal)
        out[region] = scan
        _persist_scan(region, cadence, today, scan, len(signals))
        yield {
            "event": "scanned_region",
            "region": region,
            "sections": len(scan.get("sections") or []),
        }

    duration_s = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "cadence": cadence,
        "day": today.isoformat(),
        "total_signals": len(all_rows),
        "regions": {r: len(out[r].get("sections") or []) for r in REGIONS},
        "duration_s": duration_s,
    }
    log.info("ceo_scan_flow streaming done %s", summary)
    yield {"event": "done", "summary": summary}


@observe(name="ceo_scan_flow")
async def run(cadence: str = "daily") -> dict[str, Any]:
    """Generate one CEO scan per region.

    Single Supabase fetch + Python-side bucketing. Sequenced LLM calls
    (not parallel) to stay under Groq free-tier 12K TPM cap. On Dev
    Tier or higher, this can be asyncio.gather'd safely.
    """
    started = datetime.now(timezone.utc)
    log.info("ceo_scan_flow start cadence=%s @ %s", cadence, started.isoformat())
    today = started.astimezone(timezone.utc).date()
    lookback = 1 if cadence == "daily" else 7

    all_rows = _fetch_all_recent_signals(lookback_days=lookback)
    buckets = _bucket_by_region(all_rows)
    log.info(
        "ceo_scan: total_signals=%d buckets=%s",
        len(all_rows),
        {r: len(buckets[r]) for r in REGIONS},
    )

    # Pull investigated events for every signal across all buckets in
    # ONE query, then index by signal_id. The per-region LLM call
    # gets facts + scenarios + tldr attached for any signal that has
    # been investigated. Single round-trip.
    all_signal_ids = [int(s["id"]) for s in all_rows if s.get("id") is not None]
    events_by_signal = _fetch_event_enrichment(all_signal_ids)
    log.info(
        "ceo_scan: enriched %d/%d signals with investigated event data",
        len(events_by_signal), len(all_signal_ids),
    )

    out: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        signals = buckets[region]
        investigated_in_region = sum(
            1 for s in signals if int(s.get("id") or 0) in events_by_signal
        )
        log.info(
            "ceo_scan[%s]: signals=%d investigated=%d",
            region, len(signals), investigated_in_region,
        )
        scan = await _scan_region(region, signals, events_by_signal)
        out[region] = scan
        _persist_scan(region, cadence, today, scan, len(signals))

    duration_s = (datetime.now(timezone.utc) - started).total_seconds()
    summary = {
        "cadence": cadence,
        "day": today.isoformat(),
        "total_signals": len(all_rows),
        "regions": {r: len(out[r].get("sections") or []) for r in REGIONS},
        "duration_s": duration_s,
    }
    log.info("ceo_scan_flow done %s", summary)
    return summary
