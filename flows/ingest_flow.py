"""Ingest flow: scout \u2192 persist raw \u2192 score \u2192 persist scored \u2192 promote to signals.

Week-1 scope \u2014 gnews only, title-hash dedup in signals.
Later weeks add more scouts + pgvector dedup.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.analysts import classifier
from agents.scouts import (
    ad_press_scout,
    architecture_scout,
    au_building_scout,
    au_press_scout,
    bdi_scfi_scout,
    cbuae_circulars_scout,
    cme_fedwatch_scout,
    competitor_footprint_scout,
    # 2026-04-30 stakeholder feedback: corporate_registry_scout disabled.
    # Even after the cleanup commit (c86da59) that dropped raw Companies
    # House domains and required RE-anchor + recency verbs, the signal-
    # to-noise ratio was still poor — director appointments rarely
    # connect to actionable Sobha intel and they crowd the brief. Keep
    # the file (revertible) but stop importing.
    # corporate_registry_scout,
    dfm_disclosures_scout,
    dld_projects_scout,
    # ─── Apify scouts disabled 2026-05-11 (stakeholder cost cut) ────
    # bayut_scout, dubizzle_scout, instagram_scout, meta_ads_scout —
    # all four wrote ≤8/500 visible signals in the production slice
    # for their combined ~$15-40/mo Apify spend. The classifier
    # prompt explicitly discards routine Bayut/Dubizzle listings.
    # Instagram lifestyle posts get tagged keep=false. Meta Ad
    # Library actor's signal-to-noise was poor. Coverage of the
    # underlying intelligence (Dubai launches / Modon Hudayriyat /
    # competitor brand activity) now flows through the free
    # gnews + reddit + ad_press + ir_pages mix instead.
    # Re-enable by uncommenting both the import + the .run() line.
    # bayut_scout,
    # dubizzle_scout,
    eibor_scout,
    em_real_estate_scout,
    emart_auctions_scout,
    forum_scout,
    gdelt_scout,
    geopolitics_scout,
    github_leaks_scout,
    global_hpi_scout,
    global_prime_scout,
    gnews_scout,
    gtrends_scout,
    houston_permits_scout,
    industry_research_scout,
    # instagram_scout,
    ir_pages_scout,
    linkedin_scout,
    lme_metals_scout,
    macro_scout,
    materials_scout,
    # meta_ads_scout,
    nasdaq_dubai_scout,
    newsletters_scout,
    page_hash_scout,
    policy_wire_scout,
    proptech_releasenotes_scout,
    proptech_scout,
    proptech_vc_scout,
    reddit_scout,
    sanctions_scout,
    sec_edgar_scout,
    sukuk_scout,
    sydney_auctions_scout,
    trade_press_scout,
    trakheesi_scout,
    uae_cement_scout,
    uae_press_scout,
    us_permits_scout,
    wayback_diff_scout,
    youtube_scout,
)
from flows import investigation_flow
from settings import settings
from tools import freshness, supabase_tool as sb
from tools.nvidia_llm import embed
from tools.observability import observe

log = logging.getLogger(__name__)


DEV_MATCHERS: list[tuple[str, list[str]]] = [
    # Sobha is FIRST so a co-mention like "Emirates NBD partners with
    # Sobha and Dubai Holding" tags as sobha (not dubaih) — the
    # dashboard's developer radar then filters sobha out (we're not a
    # peer to ourselves).
    ("sobha", ["sobha"]),
    ("emaar", ["emaar"]),
    ("damac", ["damac"]),
    ("aldar", ["aldar"]),
    ("nakheel", ["nakheel"]),
    ("modon", ["modon"]),
    ("dubaih", ["dubai holding", "meraas", "dubai properties"]),
    ("binghatti", ["binghatti"]),
    ("azizi", ["azizi"]),
    ("ellington", ["ellington"]),
    ("omniyat", ["omniyat"]),
    ("deyaar", ["deyaar"]),
]


def _dev_slug(text: str) -> str | None:
    t = (text or "").lower()
    for slug, needles in DEV_MATCHERS:
        if any(n in t for n in needles):
            return slug
    return None


async def _promote_to_signals(
    raw: dict[str, Any], scored: dict[str, Any]
) -> dict[str, Any] | None:
    """Upsert kept raw into `signals`. Returns the persisted row (or None).

    Applies age-based priority decay before persisting — content older
    than 14 days drops one tier; older than 180 days is rejected. Fixes
    stale-IG-post-ranks-high regressions where the scout pulls the
    latest N posts on a profile that hasn't posted in months.
    """
    # 2026-04-30: trust-the-source force-keep for curated regulatory
    # scouts. policy_wire / uae_press already passed a strict
    # five-term filter at scout time, so the classifier deciding
    # `keep=False` is almost always a false negative — and we lost
    # visa-threshold news this way. Force-keep these source families
    # plus the curated themed scouts whose domain whitelisting is
    # the actual filter.
    _TRUSTED_SOURCE_PREFIXES = (
        "policy_wire:", "uae_press:", "cbuae_circulars:",
        "newsletter:", "proptech:", "proptech_releasenotes:",
        "global_prime:", "em_real_estate:", "macro:", "geopolitics:",
        "materials:", "eibor:", "sukuk:", "sec_edgar:", "ir_pages:",
        "dfm_disclosures:", "nasdaq_dubai:", "uae_cement:",
    )
    src = (raw.get("source") or "")
    is_trusted = any(src.startswith(p) for p in _TRUSTED_SOURCE_PREFIXES)
    if is_trusted and not scored.get("keep"):
        scored = dict(scored)
        scored["keep"] = True
        if not scored.get("priority"):
            scored["priority"] = "low"
        log.info("force-keep trusted source: %s — %s", src,
                 (scored.get("headline") or raw.get("title") or "")[:80])

    if not scored.get("keep"):
        return None

    # Source publish date → priority decay.
    published_at = freshness.parse_published_at(raw)
    decayed_priority = freshness.decay_priority(scored.get("priority"), published_at)
    if decayed_priority is None:
        age = freshness.age_days(published_at)
        log.info(
            "freshness: rejecting stale signal (age=%s d): %s",
            age,
            (scored.get("headline") or raw.get("title") or "")[:80],
        )
        return None

    headline = scored.get("headline") or raw.get("title") or ""
    dek = scored.get("dek") or ""
    cluster_key = sb.title_hash(headline)
    dev = _dev_slug(f"{raw.get('title', '')} {raw.get('summary', '')}")

    # ── MD telemetry feedback loop (Round-2, 2026-04-29) ──
    # Bump priority when (dev_slug, category) matches a high-engagement
    # pair from the MD's recent click history. Loaded once per ingest
    # run and cached for cheap lookup. Fail-soft if the table is empty
    # or unavailable.
    decayed_priority = _apply_md_feedback_boost(
        decayed_priority, dev, scored.get("category"),
    )
    # Region resolution priority:
    #   1. Scout stamped it inside raw_json (gdelt knows sourcecountry,
    #      region-tuned gnews queries stamp at scout time)
    #   2. Classifier inferred it from headline + dek
    #   3. Null (dashboard groups under 'other' as fallback)
    rj = raw.get("raw_json") if isinstance(raw.get("raw_json"), dict) else {}
    region = rj.get("region") or scored.get("region") or None
    country_code = rj.get("country_code") or scored.get("country_code") or None
    # 2026-04-30 audit fix: when classifier + scout both missed the
    # region but dev_slug is a UAE developer, default to dubai (or
    # abu_dhabi for Aldar/Modon). Previously these landed in "Others"
    # which surfaced UAE-developer content in the wrong region tab.
    if not region and dev:
        DEV_REGION_DEFAULT = {
            "sobha":     ("dubai", "AE"),
            "emaar":     ("dubai", "AE"),
            "damac":     ("dubai", "AE"),
            "nakheel":   ("dubai", "AE"),
            "binghatti": ("dubai", "AE"),
            "azizi":     ("dubai", "AE"),
            "ellington": ("dubai", "AE"),
            "omniyat":   ("dubai", "AE"),
            "deyaar":    ("dubai", "AE"),
            "dubaih":    ("dubai", "AE"),
            "aldar":     ("abu_dhabi", "AE"),
            "modon":     ("abu_dhabi", "AE"),
        }
        if dev in DEV_REGION_DEFAULT:
            region, country_code = DEV_REGION_DEFAULT[dev]
            log.info("region inferred from dev_slug=%s → %s/%s", dev, region, country_code)
    # Category resolution priority (mirrors the region pattern, 2026-04-29):
    #   1. Scout-stamped `theme` or `category_hint` in raw_json — used by
    #      curated scouts (proptech / materials / geopolitics / global_prime
    #      / macro / eibor / sukuk / themed IG/LI/YT entries) to route
    #      directly to the right themed page without depending on the
    #      classifier's conservative bias toward legacy categories.
    #   2. Classifier-inferred category from headline + dek.
    # This is the lightweight alternative to a force-category schema migration.
    category = rj.get("theme") or rj.get("category_hint") or scored.get("category")
    # 2026-04-30 round 7: LLM-driven subcategory routing. The classifier
    # picks proptech/contech/frontier_ai/materials_cost/regulatory_visa/
    # etc. — themed-page endpoints filter on (category, subcategory).
    # Scout-stamped subcategory_hint takes precedence so curated scouts
    # can override the LLM when their domain whitelist already encodes
    # the answer (e.g. newsletters_scout AI feeds → frontier_ai).
    subcategory = rj.get("subcategory_hint") or scored.get("subcategory")
    # Embed headline + dek for semantic dedup; falls back to title-hash on failure.
    embed_text = f"{headline}. {dek}".strip(". ")
    embedding = await embed(embed_text)
    return sb.upsert_signal(
        {
            "cluster_key": cluster_key,
            "dev_slug": dev,
            "category": category,
            "subcategory": subcategory,
            "decision_tag": scored.get("decision_tag"),
            "priority": decayed_priority,
            "confidence": scored.get("confidence"),
            "headline": headline[:200],
            "dek": dek,
            "entities": scored.get("entities") or [],
            "urls": [raw["source_url"]] if raw.get("source_url") else [],
            "source_published_at": published_at.isoformat() if published_at else None,
            "novelty": _compute_novelty(dev),
            "region": region,
            "country_code": country_code,
        },
        embedding=embedding,
        # Pass the raw scout source through so upsert can tally
        # cross-source corroboration (2026-04-29).
        source=raw.get("source"),
    )


# Cap concurrent auto-investigations. NVIDIA NIM rate-limits under
# parallel load (seen in the 2026-04-24 run — 9 concurrent
# investigations caused entity_resolver to fail with empty responses,
# which cascaded into degenerate events). 2 concurrent is comfortable.
_INVESTIGATION_SEM = asyncio.Semaphore(2)


async def _fire_investigation(signal_id: int) -> None:
    """Fire-and-forget wrapper so ingest never waits on a slow investigation.

    Semaphore-gated so we never have more than 2 investigations running
    at once — shields NIM + Tavily + Apify from burst-induced failures.
    """
    async with _INVESTIGATION_SEM:
        try:
            result = await investigation_flow.run(signal_id, trigger="auto")
            log.info("auto-investigation done: %s", result)
        except Exception as e:
            log.warning("auto-investigation failed for signal %s: %s", signal_id, e)


## MD telemetry → priority feedback (Round-2, 2026-04-29).
## Cached for the duration of one ingest run.
_MD_FEEDBACK_WEIGHTS: dict[str, float] | None = None
_MD_FEEDBACK_LOADED_AT: float = 0.0
# 2026-04-30 audit-round: cache-miss race is benign — multiple parallel
# scouts hitting the cache miss window all fetch + idempotently
# overwrite. Worst case = one extra Supabase round-trip. Not worth
# refactoring the whole sync function into an async-with-lock shape.


def _load_md_feedback_weights() -> dict[str, float]:
    """Pull engagement weights for (dev_slug|category) from the telemetry
    table. Cached for 30 minutes — refreshed implicitly on the next
    ingest cycle. Returns {} on any failure (telemetry table missing,
    no clicks yet, etc.).
    """
    global _MD_FEEDBACK_WEIGHTS, _MD_FEEDBACK_LOADED_AT
    import time as _time
    now = _time.time()
    if _MD_FEEDBACK_WEIGHTS is not None and (now - _MD_FEEDBACK_LOADED_AT) < 1800:
        return _MD_FEEDBACK_WEIGHTS
    try:
        from datetime import timedelta as _td
        from datetime import datetime as _dt, timezone as _tz
        since = (_dt.now(_tz.utc) - _td(days=30)).isoformat()
        events = (
            sb.client().table("telemetry")
            .select("event,target_kind,target_id")
            .gte("created_at", since)
            .eq("user_id", "md")
            .limit(1000).execute().data or []
        )
    except Exception:
        _MD_FEEDBACK_WEIGHTS = {}
        _MD_FEEDBACK_LOADED_AT = now
        return {}
    EVT_W = {"pin": 3.0, "analyze_click": 2.0, "thread_open": 1.0,
             "bullet_source_click": 0.5, "signal_source_click": 0.5}
    sid_w: list[tuple[int, float]] = []
    for ev in events:
        if ev.get("target_kind") != "signal":
            continue
        try:
            sid = int(ev.get("target_id"))
        except Exception:
            continue
        w = EVT_W.get(ev.get("event"), 0.0)
        if w > 0:
            sid_w.append((sid, w))
    if not sid_w:
        _MD_FEEDBACK_WEIGHTS = {}
        _MD_FEEDBACK_LOADED_AT = now
        return {}
    sids = list({s for s, _ in sid_w})
    try:
        sigs = (
            sb.client().table("signals")
            .select("id,dev_slug,category")
            .in_("id", sids).execute().data or []
        )
    except Exception:
        _MD_FEEDBACK_WEIGHTS = {}
        _MD_FEEDBACK_LOADED_AT = now
        return {}
    by_id = {s["id"]: s for s in sigs}
    raw: dict[str, float] = {}
    for sid, w in sid_w:
        s = by_id.get(sid)
        if not s:
            continue
        dev = (s.get("dev_slug") or "_none").lower()
        cat = (s.get("category") or "_none").lower()
        raw[f"{dev}|{cat}"] = raw.get(f"{dev}|{cat}", 0.0) + w
    if not raw:
        _MD_FEEDBACK_WEIGHTS = {}
        _MD_FEEDBACK_LOADED_AT = now
        return {}
    max_w = max(raw.values())
    norm = {k: v / max_w for k, v in raw.items()}
    _MD_FEEDBACK_WEIGHTS = norm
    _MD_FEEDBACK_LOADED_AT = now
    log.info("[md-feedback] loaded %d weighted (dev,cat) pairs (max=%.2f raw)",
             len(norm), max_w)
    return norm


_PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3}
_RANK_TO_PRIORITY = {1: "low", 2: "medium", 3: "high"}


def _apply_md_feedback_boost(priority: str | None, dev_slug: str | None,
                              category: str | None) -> str | None:
    """Bump priority by ONE tier when this (dev, category) combo has a
    normalised engagement weight ≥0.6. Caps at 'high'. No-op if priority
    is None (i.e., already rejected by freshness decay).
    """
    if not priority:
        return priority
    weights = _load_md_feedback_weights()
    if not weights:
        return priority
    dev = (dev_slug or "_none").lower()
    cat = (category or "_none").lower()
    w = weights.get(f"{dev}|{cat}", 0.0)
    if w < 0.6:
        return priority
    rank = _PRIORITY_RANK.get(priority.lower(), 0)
    if rank == 0:
        return priority
    bumped = min(rank + 1, 3)
    if bumped > rank:
        log.info(
            "[md-feedback] boost %s → %s for (%s,%s) [w=%.2f]",
            priority, _RANK_TO_PRIORITY[bumped], dev, cat, w,
        )
        return _RANK_TO_PRIORITY[bumped]
    return priority


def _compute_novelty(dev_slug: str | None) -> float:
    """Quick novelty score: how rare are signals for this developer
    over the last 14 days?

      0 prior signals → 1.0 (very novel)
      1-2             → 0.85
      3-5             → 0.65
      6-10            → 0.45
      11+             → 0.25
    """
    if not dev_slug:
        return 0.5
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        cnt = (
            sb.client()
            .table("signals")
            .select("id", count="exact")
            .eq("dev_slug", dev_slug)
            .gte("last_seen_at", since)
            .execute()
            .count
            or 0
        )
    except Exception:
        return 0.5
    if cnt == 0:
        return 1.0
    if cnt <= 2:
        return 0.85
    if cnt <= 5:
        return 0.65
    if cnt <= 10:
        return 0.45
    return 0.25


@observe(name="ingest_flow")
async def run(force_linkedin: bool = False) -> dict[str, Any]:
    """Full ingest fan-out. Pass `force_linkedin=True` to bypass the
    linkedin_scout morning-only gate (used by /run/pipeline?force_linkedin=1
    when the user wants a one-off LinkedIn fire outside the 05:00 GST slot)."""
    started = datetime.now(timezone.utc)
    log.info("ingest_flow start @ %s (force_linkedin=%s)", started.isoformat(), force_linkedin)

    # 1. Scout (parallel fan-out). All scouts return [] on failure so one
    #    flaky source never poisons a run; return_exceptions guards against
    #    unexpected raises too. Master timeout (300s) caps any one scout
    #    that hangs without timing out internally — protects the whole
    #    pipeline from a single misbehaving network call (2026-04-29).
    async def _gather_with_master_timeout():
        return await asyncio.gather(
        gnews_scout.run(limit=settings.intel_query_limit),
        reddit_scout.run(limit=settings.intel_query_limit),
        youtube_scout.run(limit=settings.intel_query_limit),
        forum_scout.run(limit=settings.intel_query_limit),
        # ─── 4 Apify scouts disabled 2026-05-11 (stakeholder cost cut) ──
        # instagram_scout.run(limit=settings.intel_query_limit),
        # bayut_scout.run(limit=settings.intel_query_limit),
        # dubizzle_scout.run(limit=settings.intel_query_limit),
        linkedin_scout.run(limit=settings.intel_query_limit, force=force_linkedin),
        gdelt_scout.run(limit=60),
        sec_edgar_scout.run(limit=40),
        ir_pages_scout.run(limit=40),
        industry_research_scout.run(limit=30),
        policy_wire_scout.run(limit=20),
        architecture_scout.run(limit=25),
        trade_press_scout.run(limit=30),
        trakheesi_scout.run(limit=25),
        dld_projects_scout.run(limit=40),
        us_permits_scout.run(limit=30),
        au_building_scout.run(limit=5),
        houston_permits_scout.run(limit=2),
        github_leaks_scout.run(limit=20),
        emart_auctions_scout.run(limit=12),
        wayback_diff_scout.run(limit=25),
        page_hash_scout.run(limit=30),
        uae_press_scout.run(limit=30),
        # 2026-05-11 stakeholder ask: Abu Dhabi coverage uplift via
        # free-only AD-leaning RSS sources (The National AD, WAM,
        # Zawya AD, TradeArabia). No Tavily / Apify dependency.
        ad_press_scout.run(limit=60),
        # 2026-05-11 second pass: matching free-only AU scout. 12
        # AU-leaning English RSS feeds (news.com.au, ABC, AFR, SMH,
        # The Age, Domain, REA, Urban Developer, etc.) — closes the
        # AU coverage gap (the existing gnews/reddit/gdelt mix
        # produces <5 AU items/run).
        au_press_scout.run(limit=60),
        # 2026-04-28 stakeholder additions: EIBOR, geopolitics-with-RE-impact,
        # AI-PropTech / ConTech, Sydney + Melbourne auction depth.
        eibor_scout.run(limit=12),
        geopolitics_scout.run(limit=20),
        proptech_scout.run(limit=25),
        sydney_auctions_scout.run(limit=18),
        # 2026-04-28 Phase C: sukuk yields, construction materials,
        # global prime markets, macro / capital flow.
        sukuk_scout.run(limit=15),
        materials_scout.run(limit=20),
        global_prime_scout.run(limit=20),
        macro_scout.run(limit=18),
        # 2026-04-29 Round-2: demand intent (corroborated), rate-policy
        # probability, leading shipping indices, exchange disclosures,
        # sanctions tracker, competitor expansion footprint, PropTech
        # VC + YC + GitHub trending.
        # meta_ads_scout.run(limit=80),  # disabled 2026-05-11 (cost cut, 0 visible output)
        gtrends_scout.run(limit=20),
        cme_fedwatch_scout.run(limit=3),
        bdi_scfi_scout.run(limit=4),
        dfm_disclosures_scout.run(limit=18),
        sanctions_scout.run(limit=12),
        competitor_footprint_scout.run(limit=16),
        proptech_vc_scout.run(limit=18),
        # 2026-04-29 Round-3: real metals + city HPIs + NASDAQ Dubai sukuk
        # + UAE-specific cement + CBUAE circulars + PropTech changelogs
        # + corporate-registry directorship moves + emerging-markets RE.
        lme_metals_scout.run(limit=6),
        global_hpi_scout.run(limit=6),
        nasdaq_dubai_scout.run(limit=8),
        uae_cement_scout.run(limit=8),
        cbuae_circulars_scout.run(limit=10),
        proptech_releasenotes_scout.run(limit=12),
        # corporate_registry_scout.run(limit=12),  # disabled 2026-04-30 (low S/N)
        em_real_estate_scout.run(limit=16),
        # 2026-04-30 Round-4: curated newsletter feeds across AI, PropTech,
        # ConTech, RE markets, macro, geopolitics. Trust-the-source — the
        # newsletter IS the filter — so each item is stamped with a topic
        # category_hint that auto-routes to the matching themed page.
        newsletters_scout.run(limit=60),
        return_exceptions=True,
        )
    try:
        scout_results = await asyncio.wait_for(_gather_with_master_timeout(), timeout=300.0)
    except asyncio.TimeoutError:
        log.warning("ingest: master 300s timeout hit — proceeding with partial scout output")
        # Fallback: run an empty result set so promotion/scoring/brief
        # still execute. Better partial pipeline than no pipeline.
        scout_results = []
    scout_names = [
        "gnews", "reddit", "youtube", "forum", "instagram", "bayut",
        "dubizzle", "linkedin", "gdelt",
        "sec_edgar", "ir_pages", "industry_research",
        "policy_wire", "architecture", "trade_press", "trakheesi",
        "dld_projects", "us_permits", "au_building", "houston_permits",
        "github_leaks", "emart_auctions", "wayback_diff", "page_hash",
        "uae_press",
        "eibor", "geopolitics", "proptech", "sydney_auctions",
        "sukuk", "materials", "global_prime", "macro",
        # NOTE 2026-04-30: Round-2 (meta_ads…proptech_vc) is gathered
        # BEFORE Round-3 (lme_metals…em_real_estate). Names list
        # mirrors gather() order so failure logs attribute correctly.
        "meta_ads", "gtrends", "cme_fedwatch", "bdi_scfi",
        "dfm_disclosures", "sanctions", "competitor_footprint", "proptech_vc",
        "lme_metals", "global_hpi", "nasdaq_dubai", "uae_cement",
        "cbuae_circulars", "proptech_releasenotes",
        # corporate_registry removed 2026-04-30 (disabled in gather above)
        "em_real_estate",
        "newsletters",
    ]
    raws: list[dict[str, Any]] = []
    for name, result in zip(scout_names, scout_results):
        if isinstance(result, Exception):
            log.warning("scout %s raised: %s — skipping", name, result)
            continue
        raws.extend(result)
    log.info(
        "scouts yielded %s total=%d",
        " ".join(
            f"{n}={0 if isinstance(r, Exception) else len(r)}"
            for n, r in zip(scout_names, scout_results)
        ),
        len(raws),
    )

    # 2. Persist raw (upsert on dedup_key so re-runs are idempotent).
    inserted = sb.insert_intel_raw(raws)
    # Supabase returns upserted rows; filter to unprocessed for scoring.
    unprocessed = [r for r in inserted if not r.get("processed")]
    log.info("raw upserted=%d unprocessed=%d", len(inserted), len(unprocessed))

    # 3. Score.
    scored_rows = await classifier.score_batch(unprocessed)
    log.info("scored %d items (kept=%d)", len(scored_rows),
             sum(1 for s in scored_rows if s.get("keep")))

    # 4. Persist scored + promote kept ones into signals. Collect signal
    #    ids for every kept priority=high item so we can auto-investigate.
    raw_by_id = {r["id"]: r for r in unprocessed}
    processed_ids: list[int] = []
    high_priority_signal_ids: list[int] = []
    for s in scored_rows:
        try:
            # region + country_code live on the in-memory dict for
            # the promotion step but aren't columns on intel_scored,
            # so strip them before insert.
            scored_persisted = {k: v for k, v in s.items()
                                if k not in ("region", "country_code")}
            sb.insert_intel_scored(scored_persisted)
        except Exception as e:
            log.warning("insert intel_scored failed for raw_id=%s: %s", s.get("raw_id"), e)
            continue
        raw = raw_by_id.get(s["raw_id"])
        if raw:
            try:
                promoted = await _promote_to_signals(raw, s)
            except Exception as e:
                log.warning("promote failed for raw_id=%s: %s", s.get("raw_id"), e)
                promoted = None
            if (
                promoted
                and (s.get("priority") or "").lower() == "high"
                and promoted.get("id")
            ):
                high_priority_signal_ids.append(int(promoted["id"]))
            processed_ids.append(raw["id"])

    sb.mark_raw_processed(processed_ids)

    # 5. Auto-fire investigation on every priority=high signal.
    #    asyncio.create_task → fire-and-forget so ingest returns fast.
    for sid in high_priority_signal_ids:
        asyncio.create_task(_fire_investigation(sid))

    finished = datetime.now(timezone.utc)
    return {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "raw_scouted": len(raws),
        "raw_new": len(unprocessed),
        "scored": len(scored_rows),
        "kept": sum(1 for s in scored_rows if s.get("keep")),
        "investigations_queued": len(high_priority_signal_ids),
    }
