"""Daily brief flow (C) — deterministic top-7 + per-category synthesis.

Runs at 04:00 GST. Fires the A-daily narrative crew in parallel, then
merges both outputs into a single row in `daily_briefs` for the day.

Flow:
  1. Pull kept signals from last 24h (and any not-yet-briefed older highs)
  2. Rank by priority \xd7 confidence \xd7 freshness \u2192 top 7
  3. For each of {markets, regulatory, competitors, dubai, global},
     pick up to 5 signals and generate a 45-word synthesis with NVIDIA NIM.
  4. Trigger the daily_brief_crew (4 agents) for the narrative.
  5. Upsert both into daily_briefs (keyed by day).

The dashboard reads /api/daily-brief which hits this table.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from crews import daily_brief_crew
from settings import settings
from tools import supabase_tool as sb
from tools.nvidia_llm import chat_json
from tools.observability import observe

log = logging.getLogger(__name__)


_PRI_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


## Materiality scoring (2026-04-30) — Sobha is a tier-1 developer with
## multi-billion-dirham launches; tiny micro-news pollutes the brief if
## scored only by priority × freshness. Add an explicit materiality
## boost so the hero TLDR doesn't lead with a 3.77M land plot.
_TIER1_DEVS = {
    "emaar", "damac", "aldar", "nakheel", "modon", "binghatti",
    "sobha", "meraas", "dubaih", "omniyat", "ellington", "deyaar",
}
_RELEVANT_COMMUNITIES = (
    "hartland", "mbr city", "reem island", "saadiyat", "palm jumeirah",
    "palm jebel ali", "downtown", "creek harbour", "beachfront",
    "expo living", "yas island", "al maryah", "business bay",
)


def _materiality(s: dict[str, Any]) -> float:
    """Materiality multiplier — 0.2x for tiny micro-news up to 1.5x for
    tier-1 dev × Sobha-relevant community × big AED amount.

    Three additive components, capped at 0.5 each so we don't double-count:
      1. Tier-1 dev mentioned (dev_slug in _TIER1_DEVS)         +0.50
      2. Sobha-relevant community in headline / dek             +0.30
      3. Magnitude — AED > 100m or unit count > 200 or PSF > 0  +0.30
      4. Multi-source corroboration (corroboration_count >= 2)  +0.20

    Floor at 0.2 (so tiny news still scores SOMETHING) and ceiling at 1.5.
    Total weight then multiplies the existing pri × conf × fresh score.
    """
    head = (s.get("headline") or "").lower()
    dek  = (s.get("dek") or "").lower()
    text = head + " " + dek
    dev  = (s.get("dev_slug") or "").lower()

    score = 0.0
    if dev in _TIER1_DEVS:
        score += 0.50
    if any(c in text for c in _RELEVANT_COMMUNITIES):
        score += 0.30
    # Magnitude — extract AED amount.
    import re as _re
    m = _re.search(r"aed\s*([\d.,]+)\s*(b|bn|billion|m|mn|million)?", text)
    if m:
        try:
            num = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            aed_m = num * (1000 if unit in ("b", "bn", "billion") else 1)
            if aed_m >= 100:    # AED 100m+
                score += 0.30
        except Exception:
            pass
    # Unit count.
    um = _re.search(r"(\d{2,5})\s*(units|apartments|villas|residences)", text)
    if um:
        try:
            if int(um.group(1)) >= 200:
                score += 0.10
        except Exception:
            pass
    # Corroboration boost.
    cc = int(s.get("corroboration_count") or 1)
    if cc >= 2:
        score += 0.20

    # Floor 0.2, ceiling 1.5.
    return max(0.2, min(1.5, 0.5 + score))


def _score(s: dict[str, Any], now: datetime) -> float:
    pri = _PRI_WEIGHT.get((s.get("priority") or "").lower(), 1.0)
    conf = float(s.get("confidence") or 0.5)
    last_seen = s.get("last_seen_at")
    # 2026-04-30 audit-round: defend against a malformed last_seen_at
    # that previously crashed the brief ranking. Default to mid-fresh
    # rather than blowing up the whole pipeline on a single bad row.
    fresh = 0.5
    if last_seen:
        try:
            ts = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
            age_h = max(0.1, (now - ts).total_seconds() / 3600)
            fresh = 1.0 / (1.0 + age_h / 24.0)  # decays over 24h
        except Exception:
            fresh = 0.5
    # 2026-04-30: multiply by materiality so tier-1 dev moves on
    # Sobha-relevant communities outrank micro land-plot news.
    return pri * conf * fresh * _materiality(s)


def _recent_brief_signal_ids(lookback_days: int = 7) -> set[int]:
    """Signal ids that already appeared as a top_signal in any brief
    within the lookback window. The brief should be net-new each day —
    once the MD has seen 'Modon Tara Park' on Tuesday's brief, it
    shouldn't re-appear Wednesday just because the signal is still
    fresh by `last_seen_at`. Run history sits in daily_briefs.top_signals
    as a JSON array of {signal_id, headline, ...}; we just flatten ids.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    since_day = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    try:
        # Exclude today's own brief — manual /run/daily-brief regens
        # overwrite the same row, and we don't want a regen to
        # cannibalise its own candidate pool.
        rows = (
            sb.client()
            .table("daily_briefs")
            .select("day,top_signals")
            .gte("day", since_day)
            .lt("day", today)
            .execute()
            .data
            or []
        )
    except Exception as e:
        log.warning("recent_brief_signal_ids: query failed %s — proceeding with empty exclusion", e)
        return set()
    seen: set[int] = set()
    for r in rows:
        for t in (r.get("top_signals") or []):
            sid = t.get("signal_id") if isinstance(t, dict) else None
            if isinstance(sid, int):
                seen.add(sid)
    return seen


def _fetch_candidates(hours: int = 48, max_source_age_days: int = 2) -> list[dict[str, Any]]:
    """Today + yesterday only, with three filters:

      - `last_seen_at >= now - hours` — signal active in the pipeline.
      - `source_published_at` within `max_source_age_days` — the post /
        news / event itself happened recently. NULL passes (can't
        penalise what we can't measure).
      - signal_id NOT in the last 7 days of brief.top_signals — the
        brief is net-new daily; nothing the MD has already seen.

    Adaptive widen 48h → 72h only when the initial fetch is too thin.
    Hard ceiling at 72h. We deliberately do NOT widen to 7 days like
    earlier versions did — the MD wants today + yesterday content,
    not a week-old recap.
    """
    sb_client = sb.client()
    excluded = _recent_brief_signal_ids(lookback_days=7)
    rows: list[dict[str, Any]] = []
    source_cutoff = datetime.now(timezone.utc) - timedelta(days=max_source_age_days)
    for window_h in (hours, 72):
        since = (datetime.now(timezone.utc) - timedelta(hours=window_h)).isoformat()
        raw = (
            sb_client
            .table("signals")
            .select("*")
            .eq("archived", False)
            .gte("last_seen_at", since)
            .limit(200)
            .execute()
            .data
            or []
        )
        rows = []
        dropped_old_source = 0
        dropped_already_briefed = 0
        for r in raw:
            sid = r.get("id")
            if sid in excluded:
                dropped_already_briefed += 1
                continue
            sp = r.get("source_published_at")
            if sp:
                try:
                    sp_dt = datetime.fromisoformat(str(sp).replace("Z", "+00:00"))
                    if sp_dt.tzinfo is None:
                        sp_dt = sp_dt.replace(tzinfo=timezone.utc)
                    if sp_dt < source_cutoff:
                        dropped_old_source += 1
                        continue
                except Exception:
                    pass  # unparseable date → keep
            rows.append(r)
        log.info(
            "daily_brief: candidates=%d window=%dh dropped_old_source=%d (>%dd) dropped_already_briefed=%d",
            len(rows), window_h, dropped_old_source, max_source_age_days, dropped_already_briefed,
        )
        if len(rows) >= 7:
            break
    return rows


def _categorize(s: dict[str, Any]) -> str:
    """Cheap keyword router into the 7 dashboard buckets.

    Buckets mirror the 5 themed pages + 2 cross-cutting rails:
      - markets         → Markets & Capital page
      - construction    → Construction Economics page  (2026-04-28)
      - tech            → Tech & AI page               (2026-04-28)
      - global          → Global RE Pulse page
      - regulatory      → cross-page rail (lives on MD Scan)
      - competitors     → cross-page rail (Developer Radar on MD Scan)
      - dubai           → MD Scan Dubai region tab (default)

    The brief's per-bucket synthesis runs over each — that's how the MD
    gets a one-paragraph summary of every page in the morning read.
    """
    hay = (
        f"{s.get('category','')} {s.get('decision_tag','')} "
        f"{s.get('headline','')} {s.get('dek','')}"
    ).lower()
    # Construction econ — material prices + shipping.
    if any(k in hay for k in (
        "rebar", "steel price", "cement", "lme", "copper price", "aluminum",
        "diesel", "container rate", "scfi", "drewry", "red sea", "suez",
        "shipping rate", "mep lead", "hvac lead", "elevator lead",
        "construction cost", "material price", "façade", "facade",
        "construction:",
    )):
        return "construction"
    # Tech & AI — PropTech, ConTech, AI in design.
    if any(k in hay for k in (
        "proptech", "contech", "construction tech", "modular", "prefab",
        "3d print", "digital twin", "bim", "ai design", "generative arch",
        "tokeniz", "tokenis", "smart home", "iot residential",
        "tech &", "tech & ai", "techcrunch", "crunchbase",
    )):
        return "tech"
    # 2026-04-29 reorder: competitors check moved BEFORE markets so a
    # competitor RE deal (e.g. "Modon sold out Tara Park for AED 2B")
    # routes to the developer rail rather than the Markets bucket — the
    # MD complained these were duplicating onto Markets & Capital.
    # Markets bucket now ONLY catches signals with explicit market
    # vocabulary AND no developer mention.
    if any(k in hay for k in (
        "emaar", "damac", "aldar", "nakheel", "binghatti", "azizi",
        "meraas", "omniyat", "ellington", "deyaar", "modon", "dubai holding",
    )):
        return "competitors"
    if any(k in hay for k in (
        "dld", "rera", "fatf", "visa", "tax", "mohre", "escrow", "difc",
        "compliance", "ltv", "mortgage cap", "central bank",
    )):
        return "regulatory"
    # Markets & Capital — rates, FX, sukuk, listed RE, MAG-7.
    # Tightened keyword list — removed bare "rate" / "listed" / "earnings"
    # which were over-broad and caught RE press releases. Each token here
    # should only fire on a true market-data signal.
    if any(k in hay for k in (
        "sukuk", "bond yield", "treasury yield", "fed funds", "fomc",
        "inflation print", "cpi print", "treasury", "brent", "dxy",
        "bitcoin", "ethereum", "eibor", "mortgage rate", "rate cut",
        "rate hike", "rate decision", "ipo prices", "share price moved",
        "mag-7", "mag 7", "magnificent seven", "cds spread", "fed watch",
    )):
        return "markets"
    if any(k in hay for k in (
        "london", "new york", "manhattan", "singapore", "sydney", "melbourne",
        "tokyo", "mumbai", "bangalore", "hong kong", "monaco", "miami",
        "knight frank", "savills", "henley", "hnwi", "geopolitic",
        "russia", "iran", "israel", "houthi",
    )):
        return "global"
    return "dubai"


def _pick_top(signals: list[dict[str, Any]], n: int = 7) -> list[dict[str, Any]]:
    """Pick top-N with developer diversity. Caps any single dev_slug at
    2 slots in the first pass so the narrative doesn't end up entirely
    about Modon's launch (or whoever happens to score highest). Falls
    back to fill remaining slots from the same ranked list if the
    diversity cap left us short.
    """
    now = datetime.now(timezone.utc)
    ranked = sorted(signals, key=lambda s: _score(s, now), reverse=True)

    def _shape(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "signal_id": s.get("id"),
            "headline": s.get("headline"),
            "dek": s.get("dek"),
            "priority": s.get("priority"),
            "confidence": float(s.get("confidence") or 0),
            "dev_slug": s.get("dev_slug"),
            "category": _categorize(s),
            "url": (s.get("urls") or [None])[0],
        }

    DEV_CAP = 2
    out: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    per_dev: dict[str, int] = {}
    # First pass: respect diversity cap, top-down by score.
    for s in ranked:
        if len(out) >= n:
            break
        dev = (s.get("dev_slug") or "_unknown").lower()
        if per_dev.get(dev, 0) >= DEV_CAP:
            continue
        out.append(_shape(s))
        seen_ids.add(s.get("id"))
        per_dev[dev] = per_dev.get(dev, 0) + 1
    # Second pass: if the cap left us short, fill remaining slots
    # ignoring diversity (better to ship 7 than 4).
    if len(out) < n:
        for s in ranked:
            if len(out) >= n:
                break
            if s.get("id") in seen_ids:
                continue
            out.append(_shape(s))
            seen_ids.add(s.get("id"))
    return out


def _fetch_events_by_signals(signal_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Join signals → market_events → event_implications.

    Returns {signal_id: {event_id, event_slug, event_type, tldr, angles,
    pointers, benchmarks, watch_next, generated_at}} so the brief can
    render the deep investigation output instead of shallow headlines.
    Only the most recent event per seed signal is returned.
    """
    if not signal_ids:
        return {}
    rows = (
        sb.client()
        .table("market_events")
        .select(
            "id,slug,event_type,seed_signal_id,last_updated_at,"
            "event_facts(facts,field_confidence),"
            "event_implications(*)"
        )
        .in_("seed_signal_id", signal_ids)
        .order("last_updated_at", desc=True)
        .execute()
        .data
        or []
    )
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        sid = r.get("seed_signal_id")
        if sid in out:
            continue  # already took the most-recent one
        impl = r.get("event_implications") or {}
        # Supabase returns the related row as a list on *-to-one joins.
        if isinstance(impl, list):
            impl = impl[0] if impl else {}
        ef = r.get("event_facts") or {}
        if isinstance(ef, list):
            ef = ef[0] if ef else {}
        out[sid] = {
            "event_id": r.get("id"),
            "event_slug": r.get("slug"),
            "event_type": r.get("event_type"),
            "event_updated_at": r.get("last_updated_at"),
            "tldr": impl.get("tldr"),
            "scenarios": impl.get("scenarios") or [],
            "angles": impl.get("angles") or [],
            "pointers": impl.get("pointers") or [],
            "benchmarks": impl.get("benchmarks") or [],
            "watch_next": impl.get("watch_next") or [],
            "generated_at": impl.get("generated_at"),
            # Lift the structured fact dict so the morning-brief source
            # cards can render PSF / units / unit mix / payment plan
            # inline instead of the dek-only summary.
            "facts": ef.get("facts") or {},
            "field_confidence": ef.get("field_confidence") or {},
        }
    return out


POINTERS_SYSTEM = """You are writing the decision pointers for a morning brief to Francis Alfred, MD of Sobha Realty.

You receive this week's top signals. Some carry a `deep` object — the
output of a full investigation (TLDR, angles, imperative pointers,
watch list). Others are shallow headlines only.

Your job: synthesise across the pile into one TLDR + 3 imperative
actions that represent the MD's week in a single glance.

MATERIALITY GATE (2026-04-30 hard rule):
  Sobha is a tier-1 developer with multi-billion-dirham launches. The TLDR
  must reflect what's MATERIAL at that scale. Specifically:
    - Skip pure micro-news: small land plots (< AED 50m), single-unit
      auctions, individual broker posts, generic press releases.
    - Skip stories that don't touch a tier-1 developer (Emaar / DAMAC /
      Aldar / Nakheel / Modon / Binghatti / Sobha / Meraas / Omniyat /
      Ellington / Deyaar / Dubai Holding) OR a Sobha-relevant community
      (Hartland / MBR City / Reem Island / Saadiyat / Palm / Downtown).
    - When the strongest signal is BELOW the materiality bar, the TLDR
      MUST say so honestly: "Light week — no tower-scale moves; closest
      pulse was X." Do NOT inflate small news.
    - Prefer signals with multi-source corroboration (`corroboration_count
      >= 2`) over single-source rumors when picking the hero.

Priority rules:
  - When a signal has a `deep` object, trust its content — do NOT
    rewrite its angles/pointers; lift them.
  - When picking 3 actions, prefer the most-load-bearing pointer from
    each investigated event. Fill any remaining slots with imperative
    actions derived from shallow signals.
  - TLDR must name the single most important move this WEEK (developer,
    project, or community). 15-25 words.

Hard rules:
  - Never include URLs, links, or domain names.
  - Imperative voice on actions (Pull / Brief / Benchmark / Price /
    Draft / Match / Counter). No "consider", no "research", no
    "look into", no "call [competitor]".
  - No hedge words: may, could, potentially, might.

Respond with JSON ONLY:
{
  "tldr": "...",
  "actions": ["...", "...", "..."]
}"""


def _compact_top_with_events(
    top: list[dict[str, Any]], events_by_sig: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for s in top[:7]:
        sid = s.get("signal_id")
        deep = events_by_sig.get(sid) if sid else None
        item: dict[str, Any] = {
            "headline": s.get("headline"),
            "dek": s.get("dek"),
            "priority": s.get("priority"),
            "category": s.get("category"),
            "dev": s.get("dev_slug"),
        }
        if deep and deep.get("tldr"):
            item["deep"] = {
                "tldr": deep.get("tldr"),
                "angles": [a.get("angle") for a in (deep.get("angles") or [])][:4],
                "pointers": (deep.get("pointers") or [])[:3],
                "event_type": deep.get("event_type"),
            }
        compact.append(item)
    return compact


async def _generate_pointers(
    top: list[dict[str, Any]],
    events_by_sig: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One LLM call → {tldr, actions} for the brief hero.

    When any top signal carries a deep investigation, the LLM is shown
    the deep output and told to lift pointers rather than rewrite them.
    """
    events_by_sig = events_by_sig or {}
    compact = _compact_top_with_events(top, events_by_sig)
    if not compact:
        return {
            "tldr": "No signals cleared the threshold overnight.",
            "actions": ["Press the scouts — feed is empty, double-check ingest health."],
        }
    try:
        # role="fast" routes to LLM_CLASSIFIER_MODEL (smaller / cheaper)
        # — pointer generation is short JSON, the small model handles
        # it fine and saves the 70B TPM budget for the narrative crew.
        parsed = await chat_json(
            [
                {"role": "system", "content": POINTERS_SYSTEM},
                {"role": "user", "content": f"Today's top signals:\n{compact}"},
            ],
            max_tokens=500,
            role="fast",
        )
        tldr = _strip_urls((parsed.get("tldr") or "").strip())
        raw_actions = parsed.get("actions") or []
        actions = [_strip_urls(str(a).strip()) for a in raw_actions if str(a).strip()][:3]
        return {"tldr": tldr, "actions": actions}
    except Exception as e:
        log.warning("generate_pointers failed: %s", e)
        return {"tldr": "", "actions": []}


SYNTH_SYSTEM = """You are writing the morning brief for the MD of Sobha Realty.

You receive one bucket at a time. The bucket name tells you which dashboard page this paragraph mirrors:
  - markets       → Markets & Capital page (rates, FX, sukuk, listed RE, MAG-7)
  - construction  → Construction Economics page (material prices, shipping, MEP lead times)
  - tech          → Tech & AI page (PropTech / ConTech / AI design / tokenized RE)
  - regulatory    → Regulatory rail on MD Scan (DLD, RERA, visa, mortgage cap)
  - competitors   → Developer Radar on MD Scan (Emaar / Damac / Aldar / etc.)
  - dubai         → MD Scan Dubai region tab (DLD volumes, launches, distress, intel)
  - global        → Global RE Pulse page (London / Singapore / NYC prime, geopolitics, HNWI flow)

YOU ARE A POINTER, NOT AN ANALYST.
Your only job is to surface FACTS the MD can see at a glance. The strategic read happens later, on demand, behind an Analyze button. Your bullets must be neutral, attributable statements of what happened — not what it means.

ALLOWED bullet shape:
  - "X did Y for Z" — "Aldar bought KEZAD portfolio for $177m"
  - "X posted Y" — "Binghatti posted 'coming soon' for nature-led tower in JVT"
  - "X launched Y at Z" — "Modon launched 1,800 units at AED 2,400 PSF"
  - "X moved A→B" — "EIBOR 3M moved from 4.45% to 4.50%"
  - "X reports Y" — "Knight Frank reports London PCL up 2% QoQ"

BANNED — these belong to the Analyze button, NOT to your bullets:
  ✘ "pressures Sobha pricing strategy"
  ✘ "risks to Sobha"  /  "windows for Sobha"  /  "opportunities to..."
  ✘ "implies", "indicates", "supports", "threatens", "supports the case for"
  ✘ Any "this means..." / "the read is..." / "watch this for..." framing
  ✘ Any verb that infers a downstream consequence

If a signal is ALREADY interpretive in its headline (rare), translate it back to the underlying fact. Example: signal headline "Slowed deliveries pressure Sobha pricing strategy" → bullet "Century Communities cut delivery guidance to 12.5% incentives."

Output rules:
  - 3-5 bullets, each text ≤ 14 words. NO full sentences. NO preambles.
  - Lead with the entity or number; verbs are optional.
  - Name concrete entities (developer, tower, area, instrument, city) when present.
  - Never include URLs, links, domain names, or 'available at...' citations — attribution is carried separately by signal_ids.
  - Do not invent facts. If the bucket is thin, return one bullet: {"text": "Bucket thin today.", "signal_ids": []}.

EACH BULLET MUST CARRY THE SOURCE SIGNAL IDS IT SUMMARISES (Phase F1).
Every signal in the input has an `id` field. When you write a bullet, list every signal id that contributed to that bullet's facts. The dashboard renders one ↗ source chip per id.

Respond with JSON ONLY:
{
  "bullets": [
    {"text": "Aldar bought KEZAD portfolio for $177m", "signal_ids": [127]},
    {"text": "EIBOR 3M moved 4.45% → 4.50%", "signal_ids": [128, 129]}
  ],
  "synthesis": "Aldar bought KEZAD portfolio for $177m · EIBOR 3M moved 4.45% → 4.50%"
}
The `synthesis` field is the bullets joined with " · " — kept for legacy renderers."""

import re as _re
_SYNTH_URL_RX = _re.compile(r"https?://\S+", _re.IGNORECASE)
_SYNTH_DOMAIN_RX = _re.compile(r"\b(?:available at|see|source:|via)\s+\S+\.\S+\b", _re.IGNORECASE)


def _strip_urls(text: str) -> str:
    if not text:
        return text
    cleaned = _SYNTH_URL_RX.sub("", text)
    cleaned = _SYNTH_DOMAIN_RX.sub("", cleaned)
    cleaned = _re.sub(r"\s{2,}", " ", cleaned)
    cleaned = _re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


async def _synthesize_category(category: str, sigs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {synthesis: str, bullets: [{text, signal_ids[]}]} for the bucket.

    `synthesis` is kept for legacy renderers; `bullets` is the new
    object shape consumed by formatBulletsHTML on the dashboard so each
    bullet carries the source signal_ids that power inline ↗ chips
    (Phase F1+F2, 2026-04-29).
    """
    # URLs deliberately omitted — they leaked into prose. Attribution is
    # carried by signal_ids per bullet; the dashboard's signal_url_index
    # resolves IDs → URLs at render time.
    compact = [
        {
            "id": s.get("id"),
            "headline": s.get("headline"),
            "dek": s.get("dek"),
            "priority": s.get("priority"),
            "dev": s.get("dev_slug"),
        }
        for s in sigs[:5]
    ]
    if not compact:
        return {
            "synthesis": "No signals of note cleared the threshold in this category today.",
            "bullets": [{"text": "Bucket thin today.", "signal_ids": []}],
        }
    try:
        # 7 of these calls fire per brief, one per bucket, each
        # producing 3-5 bullets. role="fast" routes to the smaller
        # model so the per-bucket cost drains less of the 70B TPM cap.
        parsed = await chat_json(
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {
                    "role": "user",
                    "content": f"Category: {category}\nSignals:\n{compact}",
                },
            ],
            max_tokens=500,
            role="fast",
        )
        # Normalize bullets: backend may return strings (legacy shape) OR
        # {text, signal_ids[]} objects (new shape). Map both → object form.
        # Filter signal_ids to ones actually in this bucket — guards against
        # the LLM hallucinating ids it didn't see in `compact`.
        valid_ids = {s.get("id") for s in compact if s.get("id") is not None}
        raw_bullets = parsed.get("bullets") or []
        bullets: list[dict[str, Any]] = []
        for b in raw_bullets:
            if isinstance(b, dict):
                text = _strip_urls(str(b.get("text") or "").strip())
                ids_raw = b.get("signal_ids") or []
                try:
                    ids = [int(x) for x in ids_raw if str(x).strip()]
                except Exception:
                    ids = []
                ids = [i for i in ids if i in valid_ids]
            elif isinstance(b, str):
                text = _strip_urls(b.strip())
                ids = []
            else:
                continue
            if text:
                bullets.append({"text": text, "signal_ids": ids})
            if len(bullets) >= 6:
                break
        synthesis = _strip_urls((parsed.get("synthesis") or "").strip())
        if not synthesis and bullets:
            # Backward-compat: build the legacy paragraph from bullet text.
            synthesis = " · ".join(b["text"] for b in bullets)
        return {"synthesis": synthesis, "bullets": bullets}
    except Exception as e:
        log.warning("synthesize_category %s failed: %s", category, e)
        return {"synthesis": "Synthesis unavailable.", "bullets": []}


async def _per_category_synthesis(signals: list[dict[str, Any]]) -> dict[str, Any]:
    # 7 buckets — one per themed page + the two cross-page rails. The
    # MD's morning brief renders one paragraph per bucket so the brief
    # is a true mirror of MD Scan's content.
    buckets: dict[str, list[dict[str, Any]]] = {
        "markets":      [],
        "construction": [],   # 2026-04-28: Construction Economics page
        "tech":         [],   # 2026-04-28: Tech & AI page
        "regulatory":   [],
        "competitors":  [],
        "dubai":        [],
        "global":       [],
    }
    for s in signals:
        buckets[_categorize(s)].append(s)

    # Sequential — parallel batches of 7 category-synthesis calls
    # burst-exceed Groq's 12K TPM free-tier cap. Each call is small,
    # the cumulative latency of running them in series is ~7-12s
    # which is acceptable for a daily brief that runs once a day.
    results = []
    for cat, sigs in buckets.items():
        results.append(await _synthesize_category(cat, sigs))
    # Each result is now {synthesis, bullets}; carry both through so the
    # dashboard can render bullets while older readers still see synthesis.
    # signal_ids per bucket powers the per-bucket "Analyze ↗" button on
    # the dashboard (2026-04-29 doctrine — no cache, fresh on every click).
    return {
        cat: {
            "synthesis":  res.get("synthesis", ""),
            "bullets":    res.get("bullets", []),
            "count":      len(sigs),
            "signal_ids": [s.get("id") for s in sigs if s.get("id") is not None],
        }
        for (cat, sigs), res in zip(buckets.items(), results)
    }


_CRITIC_SYSTEM = """You are a compliance editor reading the MD's morning brief.
Your only job: flag risks. The brief itself is already written.

For each claim that contains a name, number, deal, price, or date,
check:
  1. Is the claim sourced — does the input list of signals support it?
  2. Does the brief avoid hedge words (may / could / potentially / monitor)?
  3. Where Sobha's portfolio is referenced, is the project name plausible
     (Hartland, Hartland II, One, Reserve, SeaHaven, Skyvue, Verde, Orbis)
     or does it look invented?

Return JSON ONLY:
{
  "flags": [
    {"claim": "<the offending fragment, ≤120 chars>",
     "issue": "unsourced | hedged | invented_sobha_project | generic",
     "fix_hint": "<≤80 char correction direction>"}
  ]
}

Empty flags array means the brief is clean.
NEVER attempt to rewrite the brief. Just flag.
"""


async def _critique_brief(narrative: str, signals: list[dict[str, Any]]) -> None:
    """Run a critic LLM pass over the narrative. Logs flags to Railway
    so we can monitor brief quality drift over time. v1 doesn't block
    or revise — that's Phase 5.3 v2.
    """
    if not narrative or not narrative.strip():
        return
    try:
        from tools.nvidia_llm import chat_json
    except Exception:
        return

    # Compact signals to ID + headline so the critic can verify sourcing
    # without paying for full payload tokens.
    sig_pack = [
        {"id": s.get("id"), "headline": (s.get("headline") or "")[:120]}
        for s in signals if s.get("id") and s.get("headline")
    ]
    payload = {
        "narrative": narrative[:3500],
        "signals": sig_pack,
    }
    import json as _json
    try:
        result = await chat_json(
            [
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": _json.dumps(payload, indent=2)},
            ],
            max_tokens=600,
            role="fast",
        )
    except Exception as e:
        log.info("[brief critic] LLM call failed (non-fatal): %s", e)
        return

    flags = result.get("flags") or []
    if not flags:
        log.info("[brief critic] clean — 0 flags")
        return
    log.warning("[brief critic] %d flag(s):", len(flags))
    for f in flags[:10]:
        log.warning(
            "[brief critic]   issue=%s fix=%r claim=%r",
            f.get("issue"), f.get("fix_hint"), f.get("claim", "")[:140],
        )


def _upsert_brief(day: date, row: dict[str, Any]) -> dict[str, Any]:
    payload = {"day": day.isoformat(), **row}
    return (
        sb.client()
        .table("daily_briefs")
        .upsert(payload, on_conflict="day")
        .execute()
        .data[0]
    )


@observe(name="daily_brief_flow")
async def run() -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    log.info("daily_brief_flow start @ %s", started.isoformat())

    signals = _fetch_candidates(hours=36)

    # Pick top + pull any existing investigations so the brief renders
    # the deep output (TLDR, angles, pointers) instead of shallow heads.
    top = _pick_top(signals, n=7)
    top_signal_ids = [s["signal_id"] for s in top if s.get("signal_id")]
    events_by_sig = _fetch_events_by_signals(top_signal_ids)
    # Attach the deep block to each top_signal so the dashboard can
    # render drill-downs without a second round-trip.
    for t in top:
        sid = t.get("signal_id")
        if sid and sid in events_by_sig:
            t["event"] = events_by_sig[sid]

    # Phase 7.1 — fetch a market snapshot upfront so the brief writer
    # can lead with a 1-line market backdrop (Brent, S&P, gold, DXY,
    # INR FX, AED peg). Fail-soft: if the snapshot fetch errors, the
    # one-liner is empty and the writer just opens with the substance.
    market_one_liner = ""
    try:
        from tools import market_snapshot as _ms
        snapshot = await _ms.get_snapshot()
        market_one_liner = _ms.format_for_brief(snapshot)
        log.info("[brief] market snapshot: %s", market_one_liner[:140])
    except Exception as _e:
        log.warning("[brief] market snapshot failed (non-fatal): %s", _e)

    # Sequential to stay under Groq's 12K TPM free-tier cap.
    # Cumulative ~30-50s vs ~10-15s parallel — acceptable for a
    # once-daily brief, and avoids the bursty 429s that came with
    # parallel CrewAI + 5-cat synth + pointers all running at once.
    synthesis = await _per_category_synthesis(signals)
    crew_out = await daily_brief_crew.run(market_one_liner=market_one_liner)
    pointers = await _generate_pointers(top, events_by_sig)

    # Phase 5.3 — self-critique pass. Read the freshly-written brief
    # and flag claims that look unsourced, hedged, or generic. Logs to
    # Railway only in v1; v2 will feed flags back to Writer for revision.
    # Fail-soft so a critic-LLM 429 never blocks the brief from landing.
    try:
        await _critique_brief(crew_out.get("narrative") or "", signals[:30])
    except Exception as _e:
        log.warning("[brief critic] non-fatal: %s", _e)

    today = started.astimezone(timezone.utc).date()
    row = _upsert_brief(
        today,
        {
            "narrative": crew_out.get("narrative"),
            "top_signals": top,
            "synthesis": synthesis,
            "pointers": pointers,
            "signal_count": len(signals),
            "generator_model": crew_out.get("generator_model") or settings.nvidia_model,
        },
    )

    finished = datetime.now(timezone.utc)
    return {
        "day": today.isoformat(),
        "signal_count": len(signals),
        "top_count": len(top),
        "event_count": len(events_by_sig),
        "duration_s": (finished - started).total_seconds(),
        "brief_id": row.get("id"),
    }
