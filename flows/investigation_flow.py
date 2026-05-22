"""Investigation flow — orchestrates seed → event_* tables.

Triggered:
  - Automatically on every newly-promoted priority=high signal
    (fire-and-forget from ingest_flow so ingest doesn't block)
  - Manually via POST /investigate/{signal_id} from the dashboard

Steps:
  1. Load seed signal from Supabase
  2. Resolve entities (cheap single NIM call)
  3. Spider: fan out across Tavily + Apify-YT + existing signals
  4. Extract facts (NIM)
  5. Write implications (NIM)
  6. Persist market_events + event_evidence + event_facts + event_implications
  7. Return summary dict

Langfuse-traced via @observe so every investigation shows up as a
single trace with input/output pairs for each step.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from agents.investigators import entity_resolver, spider
from crews import investigation_crew
from tools import evidence_typing, supabase_tool as sb
from tools.observability import observe

log = logging.getLogger(__name__)


def _slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] or "event"


def _load_signal(signal_id: int) -> dict[str, Any] | None:
    rows = (
        sb.client()
        .table("signals")
        .select("*")
        .eq("id", signal_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    return rows[0] if rows else None


def _upsert_event(
    *,
    seed_signal_id: int,
    entities: dict[str, Any],
    seed: dict[str, Any],
    facts: dict[str, Any],
) -> dict[str, Any]:
    """Upsert the market_event row.

    Identity resolution order:
      1. Existing event with same seed_signal_id → re-investigation,
         update in place (preserves slug + first_seen_at).
      2. Existing event with same slug → cross-signal dedup.
      3. Otherwise insert a new event.

    The seed_signal_id check fixes the bug where re-investigating the
    next day created a brand-new event because the slug fallback
    included today's date.
    """
    now = datetime.now(timezone.utc).isoformat()
    facts_dict = facts.get("facts") or {}
    project = facts_dict.get("project") or entities.get("project") or ""
    dev = facts_dict.get("developer") or entities.get("developer") or ""
    event_type = (facts_dict.get("event_type") or entities.get("event_type") or "other").strip().lower()

    sb_client = sb.client()

    # 1. Re-investigation lookup — same seed signal must update its
    #    own existing event regardless of how the slug shifted.
    by_seed = (
        sb_client.table("market_events")
        .select("*")
        .eq("seed_signal_id", seed_signal_id)
        .order("first_seen_at", desc=False)
        .limit(1)
        .execute()
        .data
        or []
    )
    if by_seed:
        existing = by_seed[0]
        # Keep the original slug so dashboards stay deep-linkable.
        payload = {
            "event_type": event_type or existing.get("event_type"),
            "developer_slug": entities.get("developer_slug") or existing.get("developer_slug"),
            "project_name": project or existing.get("project_name"),
            "location": facts_dict.get("location") or entities.get("location") or existing.get("location"),
            "event_date": facts_dict.get("event_date") or entities.get("event_date") or existing.get("event_date"),
            "confidence": entities.get("confidence") or existing.get("confidence") or 0.5,
            "headline": seed.get("headline") or existing.get("headline"),
            "summary": seed.get("dek") or existing.get("summary"),
            "last_updated_at": now,
        }
        return (
            sb_client.table("market_events")
            .update(payload)
            .eq("id", existing["id"])
            .execute()
            .data[0]
        )

    # 2. Slug-based dedup for events from a different seed but same
    #    developer + project + day.
    date_part = (facts_dict.get("event_date") or entities.get("event_date") or now[:10])
    slug = _slugify(f"{dev}-{project}-{date_part}")
    by_slug = (
        sb_client.table("market_events")
        .select("*")
        .eq("slug", slug)
        .limit(1)
        .execute()
        .data
        or []
    )
    payload = {
        "slug": slug,
        "event_type": event_type,
        "developer_slug": entities.get("developer_slug"),
        "project_name": project or None,
        "location": facts_dict.get("location") or entities.get("location"),
        "event_date": facts_dict.get("event_date") or entities.get("event_date"),
        "status": "ready",
        "confidence": entities.get("confidence") or 0.5,
        "headline": seed.get("headline"),
        "summary": seed.get("dek"),
        "seed_signal_id": seed_signal_id,
        "last_updated_at": now,
    }
    if by_slug:
        return (
            sb_client.table("market_events")
            .update(payload)
            .eq("id", by_slug[0]["id"])
            .execute()
            .data[0]
        )
    payload["first_seen_at"] = now
    return sb_client.table("market_events").insert(payload).execute().data[0]


def _persist_evidence(event_id: int, evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        return
    rows = [
        {
            "event_id": event_id,
            "source": e.get("source"),
            "source_url": e.get("source_url"),
            "title": (e.get("title") or "")[:500],
            "excerpt": (e.get("excerpt") or "")[:2000],
            "raw_json": e.get("raw_json") or {},
            "evidence_type": evidence_typing.classify(e.get("source"), e.get("source_url")),
        }
        for e in evidence
    ]
    sb.client().table("event_evidence").insert(rows).execute()


# Tokens that don't carry project identity. 'Places', 'Residences',
# etc. ARE kept because they distinguish branded-residence projects
# from villa communities for the same developer.
_PROJECT_STOPWORDS = {
    "by", "the", "a", "an", "of", "for", "at", "in", "on",
    "and", "&", "-",
}


def _slug_part(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:60]


def _project_tokens(dev_slug: str, project_name: str) -> frozenset[str]:
    """Token bag for project-name similarity matching."""
    normalized = re.sub(r"[^a-z0-9\s]+", " ", (project_name or "").lower())
    return frozenset(
        t for t in normalized.split()
        if t and t not in _PROJECT_STOPWORDS and t != dev_slug and len(t) > 1
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _project_fingerprint(dev_slug: str, project_name: str) -> str:
    """Stable fingerprint for primary-key-ish lookups (used as a slug
    suffix). Jaccard similarity is the actual dedup mechanism downstream."""
    tokens = sorted(_project_tokens(dev_slug, project_name))
    return f"{dev_slug}--" + "-".join(tokens[:6])


def _is_branding_partnership(nature: str | None, partners: list | None) -> bool:
    """More forgiving than exact match — Llama returns 'branding',
    'branding partnership', 'brand licensing', 'co-branding', etc.
    Treat any of those as a branded play if we also have at least one
    named non-developer partner.
    """
    if not partners:
        return False
    n = (nature or "").lower()
    branding_terms = ("brand", "licens", "co-brand", "white label")
    return any(term in n for term in branding_terms)


def _upsert_project(
    entities: dict[str, Any], facts: dict[str, Any]
) -> int | None:
    """Roll the event up into a canonical project row.

    Identity resolution:
      1. Exact slug match (legacy path).
      2. Fingerprint match — same developer + token-overlap on project
         name. Catches 'Mercedes-Benz Places by Binghatti' vs
         'Mercedes-Benz Places - Binghatti City'.
    """
    f = facts.get("facts") or {}
    project_name = (f.get("project") or entities.get("project") or "").strip()
    dev_slug = (entities.get("developer_slug") or "").lower().strip() or None
    dev_name = entities.get("developer") or None
    if not project_name or not dev_slug:
        return None
    slug = f"{_slug_part(dev_slug)}-{_slug_part(project_name)}"
    if not slug or slug == "-":
        return None
    fingerprint = _project_fingerprint(dev_slug, project_name)

    branded = _is_branding_partnership(f.get("partnership_nature"), f.get("partners"))
    brand_partner = (f.get("partners") or [None])[0] if branded else None

    payload: dict[str, Any] = {
        "slug": slug,
        "developer_slug": dev_slug,
        "developer_name": dev_name,
        "project_name": project_name,
        "location": f.get("location") or entities.get("location"),
        "units_total": f.get("units_total"),
        "unit_mix": f.get("unit_mix"),
        "starting_price_aed": f.get("starting_price_aed"),
        "starting_psf_aed": f.get("starting_psf_aed"),
        "payment_plan": f.get("payment_plan"),
        "handover_date": f.get("handover_date"),
        "target_segment": f.get("target_segment"),
        "partners": f.get("partners"),
        "branded": branded,
        "brand_partner": brand_partner,
        "fact_confidence": facts.get("field_confidence") or {},
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sb_client = sb.client()
    # 1. Exact slug match first
    existing = (
        sb_client.table("projects").select("*").eq("slug", slug).limit(1).execute().data
        or []
    )
    # 2. Fuzzy match via Jaccard token overlap on project_name. Catches
    #    spelling drift between LLM runs ('Mercedes-Benz Places by
    #    Binghatti' vs 'Mercedes-Benz Places - Binghatti City') without
    #    requiring exact set equality.
    if not existing:
        candidates = (
            sb_client.table("projects")
            .select("*")
            .eq("developer_slug", dev_slug)
            .limit(50)
            .execute()
            .data
            or []
        )
        new_tokens = _project_tokens(dev_slug, project_name)
        best_score = 0.0
        best_candidate = None
        for c in candidates:
            cand_tokens = _project_tokens(dev_slug, c.get("project_name") or "")
            score = _jaccard(new_tokens, cand_tokens)
            if score > best_score:
                best_score = score
                best_candidate = c
        # ≥0.5 Jaccard is a confident match for short project names.
        # 'mercedes/benz/places/city' vs 'mercedes/benz/places' = 0.75.
        if best_candidate and best_score >= 0.5:
            existing = [best_candidate]
            log.info(
                "project fuzzy-match: %r ~= %r (jaccard=%.2f)",
                project_name,
                best_candidate.get("project_name"),
                best_score,
            )

    if existing:
        prev = existing[0]
        # Merge: keep prior values for fields the new run nulled out
        merged = {k: v for k, v in payload.items() if v not in (None, [], {}, "")}
        merged["event_count"] = (prev.get("event_count") or 0) + 1
        merged["last_updated_at"] = payload["last_updated_at"]
        sb_client.table("projects").update(merged).eq("id", prev["id"]).execute()
        return prev["id"]

    payload["first_seen_at"] = datetime.now(timezone.utc).isoformat()
    payload["event_count"] = 1
    return sb_client.table("projects").insert(payload).execute().data[0]["id"]


def _persist_run(
    *,
    event_id: int,
    signal_id: int | None,
    facts: dict[str, Any],
    impl: dict[str, Any],
    evidence_count: int,
    duration_s: float,
    ok: bool,
    trigger: str,
) -> None:
    """Append one row to event_runs so we preserve how truth evolved."""
    try:
        sb.client().table("event_runs").insert(
            {
                "event_id": event_id,
                "signal_id": signal_id,
                "facts": facts.get("facts") or {},
                "implications": impl,
                "evidence_count": evidence_count,
                "unresolved_count": len(facts.get("unresolved") or []),
                "duration_s": duration_s,
                "ok": ok,
                "trigger_source": trigger,
            }
        ).execute()
    except Exception as e:
        log.warning("persist_run failed: %s", e)


def _set_event_state(event_id: int, status: str, *, bump_run_count: bool = False) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if bump_run_count:
        # Use Postgres-side increment via supabase rpc would be cleaner,
        # but a select-then-update keeps us off RPC machinery.
        try:
            row = (
                sb.client()
                .table("market_events")
                .select("run_count")
                .eq("id", event_id)
                .limit(1)
                .execute()
                .data
            )
            payload["run_count"] = (row[0].get("run_count") or 0) + 1 if row else 1
        except Exception:
            pass
    try:
        sb.client().table("market_events").update(payload).eq("id", event_id).execute()
    except Exception as e:
        log.warning("set_event_state(%s) failed: %s", status, e)


def _persist_facts(event_id: int, facts: dict[str, Any], model: str) -> None:
    payload = {
        "event_id": event_id,
        "facts": facts.get("facts") or {},
        "field_confidence": facts.get("field_confidence") or {},
        "unresolved": facts.get("unresolved") or [],
        "model": model,
    }
    sb_client = sb.client()
    existing = sb_client.table("event_facts").select("id").eq("event_id", event_id).limit(1).execute().data or []
    if existing:
        sb_client.table("event_facts").update(payload).eq("event_id", event_id).execute()
    else:
        sb_client.table("event_facts").insert(payload).execute()


def _persist_implications(event_id: int, impl: dict[str, Any], model: str) -> None:
    payload = {
        "event_id": event_id,
        "tldr": impl.get("tldr"),
        "scenarios": impl.get("scenarios") or [],
        "angles": impl.get("angles") or [],
        "pointers": impl.get("pointers") or [],
        "benchmarks": impl.get("benchmarks") or [],
        "watch_next": impl.get("watch_next") or [],
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    sb_client = sb.client()
    existing = sb_client.table("event_implications").select("id").eq("event_id", event_id).limit(1).execute().data or []
    if existing:
        sb_client.table("event_implications").update(payload).eq("event_id", event_id).execute()
    else:
        sb_client.table("event_implications").insert(payload).execute()


@observe(name="investigation_flow")
async def run(signal_id: int, *, trigger: str = "manual") -> dict[str, Any]:
    """Run a full investigation on a single seed signal id.

    `trigger` is logged to event_runs so we can see how a given event
    evolved (auto-fire vs sweep vs manual re-investigation).
    """
    started = datetime.now(timezone.utc)
    log.info("investigation_flow start signal_id=%s trigger=%s", signal_id, trigger)

    seed = _load_signal(signal_id)
    if not seed:
        return {"ok": False, "error": f"signal {signal_id} not found"}

    # 1. Entities
    entities = await entity_resolver.resolve(seed)
    log.info(
        "investigation: entities project=%r developer=%r type=%r",
        entities.get("project"),
        entities.get("developer_slug"),
        entities.get("event_type"),
    )

    # Guard against degenerate events. If entity resolution gave us
    # nothing AND the seed headline is also empty, we'd end up writing
    # an event with slug like '2026-04-24' and zero evidence. Abort
    # early so the prior event for this seed (if any) stays intact.
    if not any(
        [
            entities.get("project"),
            entities.get("developer"),
            entities.get("developer_slug"),
            entities.get("headline"),
            (seed.get("headline") or "").strip(),
        ]
    ):
        log.warning("investigation aborted: no usable entities for signal %s", signal_id)
        return {
            "ok": False,
            "signal_id": signal_id,
            "error": "entity_resolution_failed",
            "duration_s": (datetime.now(timezone.utc) - started).total_seconds(),
        }

    # 2. Spider
    evidence = await spider.gather(entities)

    # 3. Facts
    facts = await investigation_crew.extract_facts(seed, entities, evidence)

    # 4. Implications — wire Sobha portfolio context (migration 012 +
    # tools/sobha_context.py). The writer's IMPL_SYSTEM prompt + scenario
    # validator already require naming a Sobha asset/segment in the
    # `sobha_exposure` field; without real overlap data the LLM was
    # guessing or hallucinating ("Sobha pricing", "Sobha luxury segment").
    # Now it can anchor to real projects in the actual region.
    from tools import sobha_context as _sobha_ctx
    dev_slug_for_overlap = entities.get("developer_slug") or entities.get("dev_slug")
    sobha_overlap = _sobha_ctx.get_competitive_overlap(dev_slug_for_overlap)
    impl = await investigation_crew.write_implications(facts, entities, sobha_context=sobha_overlap)

    # 5. Persist — events first so run history can reference event_id
    from settings import settings
    event = _upsert_event(
        seed_signal_id=signal_id, entities=entities, seed=seed, facts=facts
    )
    event_id = event["id"]
    _set_event_state(event_id, "investigating")

    # Roll the event up into a canonical project (when project + dev
    # are known), then back-link the event to project_id.
    project_id = _upsert_project(entities, facts)
    if project_id:
        try:
            sb.client().table("market_events").update({"project_id": project_id}).eq(
                "id", event_id
            ).execute()
        except Exception as e:
            log.warning("link event→project failed: %s", e)

    _persist_evidence(event_id, evidence)
    _persist_facts(event_id, facts, settings.nvidia_model)
    _persist_implications(event_id, impl, settings.nvidia_model)

    finished = datetime.now(timezone.utc)
    duration_s = (finished - started).total_seconds()

    # Decide final state: 'conflicted' if unresolved questions are
    # accumulating across runs, else 'ready'. Compare the current
    # unresolved count to the most-recent prior run's count if any.
    cur_unresolved = len(facts.get("unresolved") or [])
    prior_unresolved: int | None = None
    try:
        rows = (
            sb.client()
            .table("event_runs")
            .select("unresolved_count")
            .eq("event_id", event_id)
            .order("run_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if rows:
            prior_unresolved = rows[0].get("unresolved_count")
    except Exception:
        pass
    # Hard-fail signal: extract_facts catches its own retries and emits
    # this sentinel when both attempts threw. Without escalating here, a
    # first-run failure (no prior_unresolved to compare against) would
    # land as 'ready' and the dashboard would render the seed headline
    # confidently next to an empty fact sheet.
    extraction_failed = "fact extraction failed" in (facts.get("unresolved") or [])
    facts_empty = not (facts.get("facts") or {})
    if extraction_failed or (facts_empty and cur_unresolved > 0):
        final_status = "conflicted"
    elif prior_unresolved is not None and cur_unresolved > prior_unresolved:
        final_status = "conflicted"
    else:
        final_status = "ready"
    _set_event_state(event_id, final_status, bump_run_count=True)

    # Append run history AFTER the persist + state update.
    _persist_run(
        event_id=event_id,
        signal_id=signal_id,
        facts=facts,
        impl=impl,
        evidence_count=len(evidence),
        duration_s=duration_s,
        ok=True,
        trigger=trigger,
    )

    out = {
        "ok": True,
        "signal_id": signal_id,
        "event_id": event_id,
        "event_slug": event["slug"],
        "event_type": event["event_type"],
        "project_id": project_id,
        "status": final_status,
        "evidence_count": len(evidence),
        "unresolved_count": cur_unresolved,
        "angles_count": len(impl.get("angles") or []),
        "pointers_count": len(impl.get("pointers") or []),
        "duration_s": duration_s,
        "tldr": impl.get("tldr"),
        "trigger": trigger,
    }
    log.info("investigation_flow done %s", out)
    return out
