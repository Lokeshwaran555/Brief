"""Cross-signal pattern detection — deterministic detectors.

Each detector takes a list of recent signals and returns 0+ pattern
matches. Patterns are programmatic findings (no LLM call) that
synthesise across multiple signals into a single observation:

  - DEV ACTIVITY SURGE — when a developer's recent signal count jumps
    materially above their 30-day baseline.
  - TOPIC CLUSTER — when ≥3 signals share a decision_tag across ≥2
    developers (capital wave, pricing pressure, etc.).
  - MICRO-MARKET CLUSTER — when ≥3 signals mention the same sub-market
    (Meydan, Saadiyat, Reem Island, Austin, Brisbane CBD) from
    different developers.
  - TECH-WORD CLUSTER — when ≥3 signals mention the same technology
    keyword (PropTech, modular, prefab, AI, blockchain).

Output shape per pattern:
  {
    "key": "dev_surge:emaar:7d",        # stable id for dedup
    "title": "≤80 char headline",
    "stance": "watch | risk | opportunity",
    "claim": "1-2 sentences — programmatic, no LLM",
    "evidence_signal_ids": [123, 124, ...],
    "evidence_count": 8,
    "region": "dubai | abu_dhabi | usa | australia | other | None",
    "metric": {...detector-specific extras...},
  }

LLM rationalisation is deliberately deferred — V1 is deterministic
to keep cost zero and output stable. Phase 4 wraps each match in
LLM-written prose with explicit Sobha implications.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)


# Sub-markets we care about — used by the geographic-cluster detector.
# Mention in headline/dek triggers a hit. Ordered most-specific-first
# so 'Reem Island' matches before generic 'Island'.
MICRO_MARKETS: list[tuple[str, str, str | None]] = [
    # (display label, regex pattern, optional region hint)
    ("Saadiyat Island",  r"\bsaadiyat\b",                   "abu_dhabi"),
    ("Yas Island",       r"\byas island\b",                 "abu_dhabi"),
    ("Reem Island",      r"\breem island\b",                "abu_dhabi"),
    ("Al Maryah Island", r"\b(al\s+maryah|maryah)\b",       "abu_dhabi"),
    ("Meydan",           r"\bmeydan\b",                     "dubai"),
    ("Downtown Dubai",   r"\bdowntown\s+dubai\b",           "dubai"),
    ("Hartland",         r"\bhartland\b",                   "dubai"),
    ("Marina",           r"\bdubai marina\b",               "dubai"),
    ("Palm Jumeirah",    r"\bpalm\s+jumeirah\b",            "dubai"),
    ("MBR City",         r"\b(mbr city|mohammed\s+bin\s+rashid)\b", "dubai"),
    ("JLT",              r"\b(jlt|jumeirah\s+lake)\b",      "dubai"),
    ("Austin",           r"\baustin\b",                     "usa"),
    ("Dallas",           r"\bdallas\b",                     "usa"),
    ("Houston",          r"\bhouston\b",                    "usa"),
    ("Brisbane",         r"\bbrisbane\b",                   "australia"),
    ("Sydney",           r"\bsydney\b",                     "australia"),
    ("Melbourne",        r"\bmelbourne\b",                  "australia"),
]
_MICRO_RX = [(label, re.compile(pat, re.IGNORECASE), region) for label, pat, region in MICRO_MARKETS]


# Technology / construction-method keywords — when 3+ signals mention
# the same one across different developers, it's a tech-adoption wave.
TECH_KEYWORDS: list[tuple[str, str]] = [
    ("Modular construction",     r"\bmodular\b"),
    ("Prefab residential",       r"\bprefab(ricated)?\b"),
    ("AI walkthrough",           r"\bai\s+(walkthrough|tour|virtual)\b"),
    ("PropTech",                 r"\bproptech\b"),
    ("Tokenisation",             r"\btokeni[sz]ation\b"),
    ("Branded residences",       r"\bbranded\s+residences\b"),
    ("Build-to-rent",            r"\bbuild[\s\-]to[\s\-]rent\b|\bbtr\b"),
    ("Co-living",                r"\bco[\s\-]?living\b"),
]
_TECH_RX = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in TECH_KEYWORDS]


def _signal_text(s: dict[str, Any]) -> str:
    return f"{s.get('headline') or ''} {s.get('dek') or ''}"


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        s = str(s).rstrip("Z")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ─── Detector 1: developer activity surge ──────────────────────────
def detect_dev_surge(
    signals_recent: list[dict[str, Any]],
    signals_baseline: list[dict[str, Any]],
    *,
    min_recent: int = 4,
    surge_ratio: float = 2.0,
) -> list[dict[str, Any]]:
    """Compare per-developer signal counts in the recent window vs a
    longer baseline window. Surge = ratio ≥ 2.0× and ≥4 recent signals.
    """
    recent_by_dev = Counter(s.get("dev_slug") for s in signals_recent if s.get("dev_slug"))
    base_by_dev = Counter(s.get("dev_slug") for s in signals_baseline if s.get("dev_slug"))
    out: list[dict[str, Any]] = []
    for dev, recent_n in recent_by_dev.items():
        if recent_n < min_recent:
            continue
        # Baseline weekly rate = baseline_count × (7 / baseline_window_days).
        # Caller passes 30-day baseline → weekly rate = base_n / 30 * 7.
        # We compare recent (assumed 7d) directly to that weekly rate.
        base_weekly = (base_by_dev.get(dev, 0) / 30.0) * 7.0 if base_by_dev else 0
        if base_weekly < 0.5:  # avoid divide-by-near-zero — anything looks like surge
            base_weekly = 0.5
        ratio = recent_n / base_weekly
        if ratio < surge_ratio:
            continue
        ids = [int(s["id"]) for s in signals_recent if s.get("dev_slug") == dev and s.get("id")]
        # Region of the surge — most common region tag among the recent set.
        regions = [s.get("region") for s in signals_recent if s.get("dev_slug") == dev and s.get("region")]
        region = Counter(regions).most_common(1)[0][0] if regions else None
        out.append(
            {
                "key": f"dev_surge:{dev}:7d",
                "title": f"{dev.title()} activity surged {ratio:.1f}× this week",
                "stance": "watch",
                "claim": (
                    f"{dev.title()} produced {recent_n} signals in the last 7 days — "
                    f"{ratio:.1f}× their 30-day baseline rate. Worth a closer look at "
                    f"what's driving the spike."
                ),
                "evidence_signal_ids": ids[:8],
                "evidence_count": recent_n,
                "region": region,
                "metric": {"recent_n": recent_n, "baseline_weekly": round(base_weekly, 2), "ratio": round(ratio, 2)},
            }
        )
    return out


# ─── Detector 2: topic cluster ─────────────────────────────────────
_INTERESTING_TAGS = {"capital", "launch", "pricing", "risk"}


def detect_topic_cluster(
    signals_recent: list[dict[str, Any]],
    *,
    min_signals: int = 3,
    min_devs: int = 2,
) -> list[dict[str, Any]]:
    """Find decision_tag values with ≥3 signals across ≥2 developers
    in the recent window. Indicates a market-wide move on that topic.
    """
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in signals_recent:
        tag = (s.get("decision_tag") or "").lower().strip()
        if not tag or tag == "none" or tag not in _INTERESTING_TAGS:
            continue
        by_tag[tag].append(s)

    out: list[dict[str, Any]] = []
    for tag, sigs in by_tag.items():
        if len(sigs) < min_signals:
            continue
        devs = {s.get("dev_slug") for s in sigs if s.get("dev_slug")}
        if len(devs) < min_devs:
            continue
        ids = [int(s["id"]) for s in sigs if s.get("id")][:10]
        regions = Counter(s.get("region") for s in sigs if s.get("region"))
        top_region = regions.most_common(1)[0][0] if regions else None
        # Hand-crafted claims by tag — keeps the prose precise without LLM.
        TAG_FRAMING = {
            "capital":  ("risk", "Multiple competitors raising capital — refinancing window may be closing or expansion narrative shifting."),
            "launch":   ("watch", "A cluster of new launches — supply pulse worth tracking against absorption rates."),
            "pricing":  ("opportunity", "Pricing/payment-plan moves across the comp set — read the tea leaves on demand softening."),
            "risk":     ("risk", "Risk-tagged signals clustering — regulatory/supply/macro pressure surfacing across the market."),
        }
        stance, framing = TAG_FRAMING.get(tag, ("watch", "Topic cluster detected."))
        out.append(
            {
                "key": f"topic:{tag}:{top_region or 'global'}",
                "title": f"{tag.title()} cluster — {len(sigs)} signals across {len(devs)} developers",
                "stance": stance,
                "claim": f"{framing} {len(sigs)} signals in the last week, spanning {', '.join(sorted(d for d in devs if d))[:120]}.",
                "evidence_signal_ids": ids,
                "evidence_count": len(sigs),
                "region": top_region,
                "metric": {"tag": tag, "dev_count": len(devs), "regions": dict(regions)},
            }
        )
    return out


# ─── Detector 3: micro-market cluster ──────────────────────────────
def detect_micro_market_cluster(
    signals_recent: list[dict[str, Any]],
    *,
    min_signals: int = 3,
    min_devs: int = 2,
) -> list[dict[str, Any]]:
    """≥3 signals mention the same sub-market across ≥2 developers."""
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    market_region: dict[str, str | None] = {}
    for s in signals_recent:
        text = _signal_text(s)
        for label, rx, region_hint in _MICRO_RX:
            if rx.search(text):
                by_market[label].append(s)
                market_region.setdefault(label, region_hint)

    out: list[dict[str, Any]] = []
    for market, sigs in by_market.items():
        if len(sigs) < min_signals:
            continue
        devs = {s.get("dev_slug") for s in sigs if s.get("dev_slug")}
        if len(devs) < min_devs:
            continue
        ids = [int(s["id"]) for s in sigs if s.get("id")][:8]
        out.append(
            {
                "key": f"micro_market:{market.lower().replace(' ', '_')}",
                "title": f"{market} — {len(sigs)} signals across {len(devs)} developers",
                "stance": "watch",
                "claim": (
                    f"{market} is heating up: {len(sigs)} signals in the last week "
                    f"from {len(devs)} developers. Concentration this fast usually "
                    f"signals a launch wave, pricing reset, or land scramble."
                ),
                "evidence_signal_ids": ids,
                "evidence_count": len(sigs),
                "region": market_region.get(market),
                "metric": {"market": market, "dev_count": len(devs)},
            }
        )
    return out


# ─── Detector 4: tech-keyword cluster ──────────────────────────────
def detect_tech_cluster(
    signals_recent: list[dict[str, Any]],
    *,
    min_signals: int = 3,
) -> list[dict[str, Any]]:
    """≥3 signals mention the same tech/method keyword."""
    by_keyword: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in signals_recent:
        text = _signal_text(s)
        for label, rx in _TECH_RX:
            if rx.search(text):
                by_keyword[label].append(s)

    out: list[dict[str, Any]] = []
    for keyword, sigs in by_keyword.items():
        if len(sigs) < min_signals:
            continue
        ids = [int(s["id"]) for s in sigs if s.get("id")][:8]
        regions = Counter(s.get("region") for s in sigs if s.get("region"))
        top_region = regions.most_common(1)[0][0] if regions else None
        out.append(
            {
                "key": f"tech:{keyword.lower().replace(' ', '_')}",
                "title": f"{keyword} mentions — {len(sigs)} signals this week",
                "stance": "opportunity",
                "claim": (
                    f"{keyword} appearing across {len(sigs)} signals indicates the "
                    f"category is reaching tipping. Worth scoping a pilot before "
                    f"competitors lock in vendor relationships."
                ),
                "evidence_signal_ids": ids,
                "evidence_count": len(sigs),
                "region": top_region,
                "metric": {"keyword": keyword, "regions": dict(regions)},
            }
        )
    return out


# ─── Top-level driver ──────────────────────────────────────────────
def detect_all(
    signals_recent: list[dict[str, Any]],
    signals_baseline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run all 4 detectors. Returns merged + dedup'd list of patterns,
    sorted by evidence_count desc.
    """
    patterns: list[dict[str, Any]] = []
    patterns.extend(detect_dev_surge(signals_recent, signals_baseline))
    patterns.extend(detect_topic_cluster(signals_recent))
    patterns.extend(detect_micro_market_cluster(signals_recent))
    patterns.extend(detect_tech_cluster(signals_recent))
    # Dedup on `key` (first wins).
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in patterns:
        k = p.get("key")
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(p)
    out.sort(key=lambda p: p.get("evidence_count", 0), reverse=True)
    return out


# ─── LLM-prose layer ──────────────────────────────────────────────
#
# Each pattern's deterministic claim is fine for diagnostics but the
# CEO wants a Sobha-specific implication, not a generic observation.
# For each detected pattern we run ONE small LLM call that produces:
#   - Why this matters to Sobha specifically (1 sentence)
#   - Suggested action verb (Pull / Brief / Benchmark / etc.)
# routes through role="fast" → LLM_CLASSIFIER_MODEL (8B-instant).
#
# Cached in-process by (pattern.key, day) so re-calls within a day
# don't burn tokens. Cache survives until process restart (Railway
# redeploy resets it; that's fine).

PROSE_SYSTEM = """You are writing a one-sentence strategic implication for Francis Alfred,
MD of Sobha Realty. You receive a market pattern detected across multiple
external signals and must produce a Sobha-flavoured 'so-what'.

ARCHITECTURE: External Subject / Sobha Anchor.
- The pattern is the SUBJECT — describes external behaviour (competitor
  surge, capital flow, segment shift, etc.). Sobha is the LENS used
  to interpret it.
- Your sobha_implication describes how that EXTERNAL pattern repositions
  Sobha's competitive frame (pricing power, supply tightness, capital
  cost, channel access). Never lead with Sobha's internal facts.

INPUT — pattern dict with: title, claim, stance (opportunity/risk/watch),
region, evidence_count, metric (detector-specific extras), plus:
  - sobha_portfolio_summary: REFERENCE FRAME — Sobha's actual projects
    (use these names verbatim — never invent)
  - sobha_competitive_overlap: when present, explicit overlap with
    the developer this pattern names — regions, communities, adjacent
    Sobha projects (including geographic_adjacencies for Tier-1
    location intelligence), segment overlap, analyst notes

OUTPUT — JSON ONLY:
{
  "sobha_implication": "1 sentence — how this external pattern shifts Sobha's competitive position. Name a Sobha asset / community / segment FROM the portfolio_summary or competitive_overlap. If the overlap block confirms no projects in the relevant region, explicitly say 'No direct Sobha exposure — watch-list only'. NEVER invent a project name not in the input.",
  "action": "Imperative verb-led action (Pull / Brief / Benchmark / Match / Counter / Draft / Publish / Commission). 1 sentence, ends in a deliverable."
}

DATA-GAP DISCIPLINE: if the pattern lacks a critical data point (e.g.
the metric is unspecified, the dev_slug isn't in the overlap map),
include in your sobha_implication: "Need: <specific question>. Delegate
to: <web_search | DLD lookup | Sobha analyst>." Never fabricate.

Rules:
- No hedging: NEVER use 'may / could / potentially / consider / monitor'
- Name competitors specifically (Emaar, DAMAC, Aldar, Mubadala, Modon, Camden, Mirvac, etc.)
- 'sobha_implication' references the EXTERNAL move's effect on Sobha's
  position; Sobha is never the grammatical subject of the implication
- 'action' must end in a deliverable (memo / brief / price-sheet /
  vendor scope) — verb-only fails ('research X' / 'monitor X')
"""


_PROSE_CACHE: dict[str, dict[str, Any]] = {}


def _prose_cache_key(pattern_key: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{today}:{pattern_key}"


def _week_start(d: datetime | None = None) -> str:
    """Monday of the given (or current) week, ISO date string."""
    d = d or datetime.now(timezone.utc)
    monday = d - timedelta(days=d.weekday())
    return monday.date().isoformat()


def persist_patterns(patterns: list[dict[str, Any]]) -> int:
    """Upsert detected patterns into market_patterns keyed by
    (key, week_start). Idempotent — re-runs within the same week
    overwrite. Defensive: if the migration hasn't applied yet, logs
    a warning and returns 0 rather than blowing up the caller.

    Returns count of rows upserted.
    """
    if not patterns:
        return 0
    week = _week_start()
    rows = []
    for p in patterns:
        key = p.get("key")
        if not key:
            continue
        rows.append({
            "key": key,
            "week_start": week,
            "title": p.get("title"),
            "stance": p.get("stance"),
            "claim": p.get("claim"),
            "region": p.get("region"),
            "evidence_count": int(p.get("evidence_count") or 0),
            "evidence_signal_ids": p.get("evidence_signal_ids") or [],
            "metric": p.get("metric") or {},
            "sobha_implication": p.get("sobha_implication"),
            "action": p.get("action"),
        })
    if not rows:
        return 0
    try:
        # Lazy import so this module stays test-importable.
        from tools import supabase_tool as sb
        sb.client().table("market_patterns").upsert(
            rows, on_conflict="key,week_start"
        ).execute()
        return len(rows)
    except Exception as e:
        msg = str(e).lower()
        if "market_patterns" in msg or "relation" in msg:
            log.warning(
                "persist_patterns: market_patterns table missing — "
                "run migration 010_market_patterns.sql. Skipping persist."
            )
        else:
            log.warning("persist_patterns failed: %s", e)
        return 0


def fetch_pattern_history(
    *,
    weeks: int = 8,
    region: str | None = None,
    pattern_key: str | None = None,
) -> list[dict[str, Any]]:
    """Pull persisted patterns for the historical view. Returns rows
    sorted week_start desc. Defensive on missing table.
    """
    try:
        from tools import supabase_tool as sb
        cutoff = (datetime.now(timezone.utc) - timedelta(weeks=weeks)).date().isoformat()
        q = (
            sb.client()
            .table("market_patterns")
            .select("*")
            .gte("week_start", cutoff)
            .order("week_start", desc=True)
            .limit(500)
        )
        if region:
            q = q.eq("region", region)
        if pattern_key:
            q = q.eq("key", pattern_key)
        return q.execute().data or []
    except Exception as e:
        log.warning("fetch_pattern_history failed: %s", e)
        return []


async def enrich_with_prose(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add `sobha_implication` + `action` to each pattern via a small
    LLM call. Uses an in-process cache keyed by (pattern_key, today)
    so dashboard polls don't re-spend tokens.

    Lazy import of chat_json so this module stays importable without
    LLM credentials configured (e.g. for unit tests).
    """
    if not patterns:
        return patterns
    try:
        from tools.nvidia_llm import chat_json
    except Exception as e:
        log.warning("enrich_with_prose: chat_json unavailable (%s) — skipping", e)
        return patterns

    # Sobha context — fetched once per call, threaded into every prose
    # enrichment so 'sobha_implication' references real assets.
    from tools import sobha_context as _sobha_ctx
    sobha_summary = _sobha_ctx.format_for_llm("summary")

    enriched: list[dict[str, Any]] = []
    for p in patterns:
        key = p.get("key") or ""
        ck = _prose_cache_key(key)
        cached = _PROSE_CACHE.get(ck)
        if cached:
            enriched.append({**p, **cached})
            continue
        try:
            # Per-pattern overlap (when a dev_slug is on the metric blob)
            metric = p.get("metric") or {}
            dev_slug = metric.get("dev_slug") or metric.get("developer_slug")
            overlap_text = _sobha_ctx.format_overlap_for_llm(dev_slug) if dev_slug else ""

            user_payload = {
                "title": p.get("title"),
                "claim": p.get("claim"),
                "stance": p.get("stance"),
                "region": p.get("region"),
                "evidence_count": p.get("evidence_count"),
                "metric": metric,
                "sobha_portfolio_summary": sobha_summary,
                "sobha_competitive_overlap": overlap_text,
            }
            import json as _json
            messages = [
                {"role": "system", "content": PROSE_SYSTEM},
                {"role": "user", "content": _json.dumps(user_payload, indent=2)},
            ]
            out = await chat_json(messages, max_tokens=300, role="fast")
            if isinstance(out, dict):
                imp = (out.get("sobha_implication") or "").strip()
                act = (out.get("action") or "").strip()
                addon = {"sobha_implication": imp, "action": act}
                _PROSE_CACHE[ck] = addon
                enriched.append({**p, **addon})
                continue
        except Exception as e:
            log.warning("enrich_with_prose[%s] failed: %s", key, e)
        # On any failure, keep the pattern without prose so the rail
        # still renders — graceful degradation.
        enriched.append(p)
    return enriched
