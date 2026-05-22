"""Investigation crew — reasoning over the spider's evidence pack.

Two LLM calls (not CrewAI — we don't need the 4-agent orchestration
machinery for well-defined steps, and each call is cheaper + easier to
debug as a direct NIM chat_json).

  Step 1: fact_extractor  → structured JSON of every field we can pin
                           down, with per-field confidence + unresolved
                           questions list.
  Step 2: implication_writer → Sobha-specific TL;DR + angles +
                              imperative pointers + what-to-watch.

Persist to event_facts + event_implications tables from the flow layer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from tools.nvidia_llm import chat_json


# ─── Scenario quality validator ─────────────────────────────────────
#
# Programmatic enforcement of the BAD-EXAMPLE block in IMPL_SYSTEM.
# The prompt asks the LLM not to hedge, but Llama drifts. Validator
# catches drift, retries with a targeted critique once, then drops
# any scenario still failing rather than persisting weak output.
_HEDGE_RX = re.compile(
    r"\b(may|could|potentially|might|consider|monitor|watch|explore|"
    r"investigate|look into|research)\b",
    re.IGNORECASE,
)
_GENERIC_ACTION_VERB_RX = re.compile(
    r"^\s*(monitor|watch|consider|explore|research|look into|investigate)\b",
    re.IGNORECASE,
)


def _validate_scenarios(scenarios: list[dict[str, Any]]) -> list[tuple[int, list[str]]]:
    """Return [(scenario_index, [reasons])] for every scenario that
    fails quality checks. Empty list = all good.

    Checks per scenario:
      - claim: ≥30 chars, no hedge words
      - what_supports: ≥20 chars
      - what_would_confirm: ≥15 chars (real falsification test)
      - what_would_reject: ≥15 chars
      - sobha_exposure: ≥15 chars (names a Sobha asset / segment)
      - action: ≥12 chars, doesn't start with generic verb,
        no hedge words
    """
    issues: list[tuple[int, list[str]]] = []
    for i, s in enumerate(scenarios or []):
        if not isinstance(s, dict):
            issues.append((i, ["scenario not an object"]))
            continue
        reasons: list[str] = []
        claim = (s.get("claim") or "").strip()
        action = (s.get("action") or "").strip()
        what_supports = (s.get("what_supports") or "").strip()
        what_confirm = (s.get("what_would_confirm") or "").strip()
        what_reject = (s.get("what_would_reject") or "").strip()
        sobha_exposure = (s.get("sobha_exposure") or "").strip()

        if len(claim) < 30:
            reasons.append("claim too short or empty")
        elif _HEDGE_RX.search(claim):
            reasons.append("claim uses hedge words (may/could/potentially/might)")
        if len(what_supports) < 20:
            reasons.append("what_supports too short or empty")
        if len(what_confirm) < 15:
            reasons.append("what_would_confirm missing — every scenario needs a falsification test")
        if len(what_reject) < 15:
            reasons.append("what_would_reject missing — every scenario needs a falsification test")
        if len(sobha_exposure) < 15:
            reasons.append("sobha_exposure missing — must name a Sobha asset / segment / channel")
        if len(action) < 12:
            reasons.append("action too short or empty")
        elif _GENERIC_ACTION_VERB_RX.match(action):
            reasons.append("action starts with generic verb (monitor/watch/consider/explore)")
        elif _HEDGE_RX.search(action):
            reasons.append("action contains hedge words")
        if reasons:
            issues.append((i, reasons))
    return issues


async def _retry_implications_with_critique(
    messages: list[dict[str, str]],
    prior_result: dict[str, Any],
    issues: list[tuple[int, list[str]]],
) -> dict[str, Any] | None:
    """One targeted retry. Sends the prior assistant response back
    along with a per-scenario critique so the LLM knows exactly which
    fields are wrong. Returns the retry's parsed JSON or None on
    failure (caller falls back to dropping bad scenarios).
    """
    critique_lines = []
    for idx, reasons in issues:
        critique_lines.append(f"  - scenario {idx + 1}: {'; '.join(reasons)}")
    critique = "\n".join(critique_lines)
    nudge = (
        "Your previous response had quality issues. Fix them and resend the "
        "FULL JSON object (same shape as the system prompt schema):\n\n"
        f"{critique}\n\n"
        "Hard rules: NEVER use 'may / could / potentially / might / consider / "
        "monitor / watch / explore / investigate'. Every scenario MUST include "
        "what_supports, what_would_confirm, what_would_reject, sobha_exposure, "
        "and an action starting with an imperative deliverable verb (Pull / "
        "Brief / Benchmark / Price / Draft / Match / Counter / Publish / "
        "Commission). Falsification fields MUST name a metric + horizon, not "
        "a vague 'sales drop'."
    )
    retry_msgs = messages + [
        {"role": "assistant", "content": json.dumps(prior_result)},
        {"role": "user", "content": nudge},
    ]
    try:
        return await chat_json(retry_msgs, max_tokens=2500)
    except Exception as e:
        log.warning("implications validator retry failed: %s", e)
        return None

log = logging.getLogger(__name__)


FACTS_SYSTEM = """You are a property-desk analyst building a structured fact sheet for
a market move in Dubai real estate. The MD (Francis Alfred, Sobha Realty)
will read your output to decide how to respond.

You receive:
  - seed signal (the headline that triggered the investigation)
  - resolved entities (developer, project, location, event_type)
  - evidence pack: a list of artifacts the spider gathered from
    search, YouTube, news, our prior signals.

EXTRACTION RULES — be aggressive, not paralyzed:
  - If a fact appears in ANY single credible source (Wikipedia,
    developer site, major news, YouTube transcript), EXTRACT IT with
    medium-to-high confidence. Do not wait for triple-source
    corroboration.
  - If the title or caption of a video/post names a figure (e.g.
    "AED 30 Billion Mega Project", "2,200 residences", "Meydan"),
    USE IT — these are first-party claims from the developer or
    from an industry reporter.
  - If sources conflict, pick the most-cited value. Log the conflict
    in `unresolved`, do NOT null the field.
  - Only leave a field null if NO source in the pack mentions it
    directly or implicitly. 'Not mentioned' ≠ 'null'; 'contradicted
    beyond reconciliation' = null.
  - Numbers matter. AED amounts, unit counts, PSFs, dates. Pull them.
  - Location should be as SPECIFIC as the evidence allows (community/
    sub-market level: Meydan, Reem Island, Downtown — not just Dubai).

Always return the full object; use null only when genuinely unknown.

Respond with JSON ONLY. Schema:
{
  "facts": {
    "project": "...",
    "developer": "...",
    "location": "...",
    "event_type": "...",
    "event_date": "YYYY-MM-DD | null",
    "launch_event": "venue / name of launch event | null",
    "units_total": int | null,
    "unit_mix": ["studio","1BR","2BR",...] | null,
    "starting_price_aed": int | null,
    "starting_psf_aed": int | null,
    "payment_plan": "60/40 post-handover | ... | null",
    "handover_date": "YYYY-QN or YYYY-MM | null",
    "broker_commission_pct": number | null,
    "partners": ["Mubadala","..."] | [],
    "partnership_nature": "equity JV | land contribution | branding | ... | null",
    "capital_structure": "sukuk / bank debt / equity / ... | null",
    "target_segment": "end-user | investor | HNW | branded residences | ... | null",
    "sales_channel": "in-house | brokers | invite-only | EOI | ... | null"
  },
  "field_confidence": {  // 0.0-1.0 per field populated above
    "<field>": 0.0
  },
  "unresolved": [
    "open question 1 (e.g. 'exact unit count conflicts between X and Y')",
    ...
  ]
}"""


IMPL_SYSTEM = """You are writing a decision brief for Francis Alfred, MD of Sobha Realty,
based on a structured fact sheet from a market-move investigation.

ARCHITECTURE: External Subject / Sobha Anchor.
- The investigated move is an EXTERNAL event (competitor launch, regulatory
  shift, capital deployment, infrastructure announcement). The SUBJECT of
  every scenario's claim is that external move + its mechanism.
- Sobha is the LENS — the exposure call, the KPI benchmark, the
  competitive position. Sobha is never the grammatical subject of a
  scenario's claim.
- The MD already knows what Sobha owns. Don't tell him. Tell him how
  the EXTERNAL move repositions Sobha's pricing / capital / supply /
  channel frame.

The MD's job is to make decisions under uncertainty. A brief that
collapses a competitor move into one deterministic conclusion ("this is
bad for Sobha") gives him false confidence. Instead, frame each move as
2-3 SCENARIOS — distinct reads of what the EXTERNAL move could mean
for Sobha's position — and let him pick.

The user payload includes a `sobha_context` block (REFERENCE FRAME) with
Sobha's actual competitive overlap with this developer: regions,
communities, adjacent Sobha projects (including `geographic_adjacencies`
for Tier-1 location intelligence — shared boundary / demand corridor),
segment overlap, and a list of live Sobha projects in the relevant
region. USE IT. Pull project names from `sobha_context.sobha_projects_in_region`
and community overlaps from `sobha_context.communities_overlap`.
If `sobha_context.loaded` is false, say so in the tldr and treat
exposure assessments as directional.

KPI LENS: read the move on standard RE KPIs — AED PSF, sold %,
absorption velocity, payment-plan length, GP%, FAR. Where the facts
provide the data, compute apples-to-apples competitor-vs-Sobha
comparisons.

DATA-GAP DISCIPLINE: when a critical data point is unknown (PSF not
disclosed, partner not yet announced, payment plan ambiguous), DO NOT
fabricate. Add to the relevant scenario's `what_would_confirm` field
the explicit gap shape: "Need: <specific question>. Delegate to:
<web_search | DLD lookup | Sobha analyst | nobody>." This is how the
brief calls for the data Sobha needs to commit.

Reference Sobha projects by name where the context provides them.
Avoid invented project names.

Respond with JSON ONLY (no prose before or after):
{
  "tldr": "...",
  "scenarios": [
    {
      "stance": "opportunity" | "risk" | "watch",
      "claim": "one-sentence read of what this move could mean",
      "what_supports": "what in the facts/evidence backs this read",
      "what_would_confirm": "what new data point would prove it true",
      "what_would_reject": "what would falsify this read",
      "sobha_exposure": "which Sobha project / segment / channel is in play",
      "action": "imperative recommended action — Pull / Brief / Benchmark / Price / Draft / Match / Counter / Publish / Commission"
    }
  ],
  "pointers": [...],
  "benchmarks": [...],
  "watch_next": [...]
}

REQUIRED SIZES:
  - scenarios:  exactly 2 or 3 items, with DISTINCT stances. Always
                include at least one of {opportunity, risk}; the third
                should typically be a 'watch' (a read that needs more
                data before it commits). Do not write three scenarios
                that all say the same thing in different words.
  - pointers:   exactly 3 items — top imperative actions across all
                scenarios, deliverable-shaped. Do NOT repeat the per-
                scenario `action` field verbatim.
  - benchmarks: 1 to 3 items
  - watch_next: 2 to 3 items
  - tldr:       15-25 words, must contain at least one concrete number
                OR proper noun from the facts

POINTERS must be actions Sobha's own teams can take (sales, land,
strategy, IR, marketing, brand). Start with imperative verbs:
  Pull / Brief / Benchmark / Price / Draft / Match / Counter / Publish / Commission.
Do NOT write pointers that require cooperation from the competitor
(e.g. "Call Binghatti to discuss" — Sobha doesn't phone rivals).
Do NOT write research-only pointers ("research", "look into",
"investigate", "explore"). Every pointer ends in a deliverable.

BENCHMARKS RULE: `competitor_value` must be either (a) a value present
in the facts object, or (b) the literal string "not disclosed in
evidence". NEVER fabricate a number. If facts are empty, set
benchmarks to [] and say so in the tldr.

Never include URLs, links, or domain names.

────────────────────────────────────────────────────────
GOOD EXAMPLE — Aldar launches studio-heavy project in Dubailand:
{
  "tldr": "Aldar's studio-heavy Dubailand launch tests low-ticket investor appetite — three reads possible before Sobha responds.",
  "scenarios": [
    {
      "stance": "risk",
      "claim": "Investor demand is rotating toward low-ticket entry product, pulling enquiries away from Sobha One's 1-2BR mix.",
      "what_supports": "Aldar concentrating on studios at scale signals a read that ticket size matters more than location at current rates.",
      "what_would_confirm": "Sobha One enquiry-to-booking conversion shifts toward smaller units in the next 14 days.",
      "what_would_reject": "Sobha One conversion mix holds steady or skews larger over the same window.",
      "sobha_exposure": "Sobha One — investor 1-2BR slice; Hartland II studios if any.",
      "action": "Pull enquiry-to-booking conversion by unit size for the last 14 days."
    },
    {
      "stance": "watch",
      "claim": "Aldar may be using studios for absorption velocity, not because studios are structurally strongest demand.",
      "what_supports": "Studio-heavy launches book fast and let developers post strong week-one absorption regardless of underlying mix demand.",
      "what_would_confirm": "Aldar's DLD first-registration pace skews studio-heavy in week 1 then plateaus.",
      "what_would_reject": "Aldar's launch sells out across all unit types within 30 days, indicating real cross-segment demand.",
      "sobha_exposure": "Strategy team's launch-mix decision for next Sobha release.",
      "action": "Benchmark Aldar's DLD first-registration mix against headline launch claim over 30 days."
    },
    {
      "stance": "opportunity",
      "claim": "If studios are being pushed because Aldar's larger-unit demand is soft, Sobha's family/end-user product stays differentiated.",
      "what_supports": "A pivot to studios at this scale is unusual for Aldar's brand; suggests larger units harder to absorb at current price.",
      "what_would_confirm": "2BR+ portal listings in the same micro-market show price cuts or longer days-on-market over 30 days.",
      "what_would_reject": "Aldar releases a 2BR+ launch at flat or higher PSF in the next 60 days.",
      "sobha_exposure": "Hartland II and One family-unit positioning; broker brief.",
      "action": "Brief sales to lean into family/end-user messaging until 2BR+ portal data turns."
    }
  ],
  "pointers": [
    "Pull enquiry-to-booking conversion by unit size for Sobha One and Hartland II for the last 14 days.",
    "Benchmark DLD first-registration mix in Dubailand against Aldar's headline launch claim over 30 days.",
    "Brief sales to lean into family/end-user messaging until 2BR+ portal pricing turns."
  ],
  "benchmarks": [
    {"metric": "studio share of launch", "competitor_value": "not disclosed in evidence", "sobha_value": "unknown — MIS not wired", "delta": "—"},
    {"metric": "starting price AED", "competitor_value": "not disclosed in evidence", "sobha_value": "unknown — MIS not wired", "delta": "—"}
  ],
  "watch_next": [
    "Aldar's DLD first-registration mix in Dubailand over the next 30 days.",
    "2BR+ portal listing discounts in the same micro-market.",
    "Broker commission tier Aldar offers in week 2-4 of the campaign."
  ]
}

────────────────────────────────────────────────────────
BAD EXAMPLE — do NOT write like this:

{
  "tldr": "Aldar launches studios in Dubailand",       // ❌ 5 words; no stakes
  "scenarios": [
    {
      "stance": "risk",
      "claim": "This may be bad for Sobha sales.",     // ❌ 'may'; vague; names no Sobha asset
      "what_supports": "Competitors launching is generally a risk.",  // ❌ generic
      "what_would_confirm": "Sales go down.",          // ❌ unfalsifiable; no metric, no horizon
      "what_would_reject": "Sales don't go down.",
      "sobha_exposure": "Sobha overall.",              // ❌ everything-and-nothing
      "action": "Monitor the situation."               // ❌ 'monitor'; no deliverable
    },
    {
      "stance": "risk",                                // ❌ duplicate stance
      "claim": "Pricing pressure on Sobha is possible.",
      ...
    }
  ]
}

Avoid every pattern above. Each scenario must commit to a specific
read, name a specific Sobha asset, propose a specific falsification
test, and end in a specific deliverable. If facts are thin, stretch
for a sharper read grounded in whatever IS known — never hedge with
"may", "could", "potentially", "might".
────────────────────────────────────────────────────────"""


# Sentinel returned to the flow when fact extraction failed or facts
# came back empty. The flow uses this to skip persisting hallucinated
# scenarios; the dashboard renders status='conflicted' and shows the
# tldr verbatim instead of an angle list.
EMPTY_IMPL: dict[str, Any] = {
    "tldr": "Investigation inconclusive — fact extraction failed; no scenarios generated.",
    "scenarios": [],
    "angles": [],
    "pointers": [],
    "benchmarks": [],
    "watch_next": [],
}


def _scenarios_to_angles(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive a backward-compat `angles[]` view from `scenarios[]`.

    `flows/daily_brief_flow.py` and the dashboard's preview cards still
    read `angles`. Keep both shapes populated until those readers are
    cut over to scenarios natively.
    """
    out: list[dict[str, Any]] = []
    for s in scenarios or []:
        stance = (s.get("stance") or "").strip()
        claim = (s.get("claim") or "").strip()
        if not claim:
            continue
        # Angle label = stance prefix so brief surfaces stay distinguishable.
        label = stance.upper() if stance else "READ"
        out.append({"angle": label, "reasoning": claim})
    return out


def _compact_evidence(evidence: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    """Compact but keep enough body for the LLM to mine facts.

    Prioritise items that have non-empty excerpts (transcripts, Wikipedia
    paragraphs, descriptions) so title-only YT thumbnails don't crowd out
    high-content items.
    """
    ranked = sorted(
        evidence,
        key=lambda e: (1 if (e.get("excerpt") or "").strip() else 0, len(e.get("excerpt") or "")),
        reverse=True,
    )
    out = []
    for e in ranked[:limit]:
        out.append(
            {
                "source": e.get("source"),
                "title": e.get("title"),
                "excerpt": (e.get("excerpt") or "")[:1000],
            }
        )
    return out


async def extract_facts(
    seed: dict[str, Any],
    entities: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "seed": {
            "headline": seed.get("headline"),
            "dek": seed.get("dek"),
        },
        "entities": entities,
        "evidence_count": len(evidence),
        "evidence": _compact_evidence(evidence),
    }
    messages = [
        {"role": "system", "content": FACTS_SYSTEM},
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]
    # Llama through NIM occasionally drops JSON mode and returns prose.
    # Two-attempt pattern: bare, then with an explicit nudge.
    try:
        return await chat_json(messages, max_tokens=2500)
    except Exception as e1:
        log.warning("extract_facts attempt 1 failed: %s — retrying", e1, exc_info=True)
    retry_messages = messages + [
        {
            "role": "user",
            "content": "Return ONLY the JSON object defined in the system prompt. No prose, no preamble, no markdown fences. Start your response with '{'.",
        }
    ]
    try:
        return await chat_json(retry_messages, max_tokens=2500)
    except Exception as e2:
        log.warning("extract_facts attempt 2 failed: %s", e2, exc_info=True)
        return {"facts": {}, "field_confidence": {}, "unresolved": ["fact extraction failed"]}


async def write_implications(
    facts: dict[str, Any],
    entities: dict[str, Any],
    sobha_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Gate: if fact extraction failed or returned an empty fact dict
    # alongside unresolved questions, generating 2-3 confident scenarios
    # over nothing is the exact failure mode this rewrite is meant to
    # eliminate. Return the empty sentinel — the flow will mark the
    # event 'conflicted' and the drawer renders the tldr verbatim.
    facts_dict = facts.get("facts") or {}
    unresolved = facts.get("unresolved") or []
    if "fact extraction failed" in unresolved or (not facts_dict and unresolved):
        log.info("write_implications skipped: extraction failed or empty facts")
        return EMPTY_IMPL

    payload = {
        "entities": entities,
        "facts": facts_dict,
        "unresolved": unresolved,
        "sobha_context": sobha_context or {"status": "MIS not yet wired"},
    }
    messages = [
        {"role": "system", "content": IMPL_SYSTEM},
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]
    # Two attempts: first with JSON mode, second with an explicit reminder.
    # Llama occasionally ignores JSON mode on the first call and returns
    # prose — retry with a direct nudge usually fixes it.
    result: dict[str, Any] | None = None
    try:
        result = await chat_json(messages, max_tokens=2500)
    except Exception as e1:
        log.warning("write_implications attempt 1 failed: %s — retrying", e1, exc_info=True)
        retry_messages = messages + [
            {
                "role": "user",
                "content": "Return ONLY the JSON object defined in the system prompt. No prose, no preamble, no markdown fences. Start your response with '{'.",
            }
        ]
        try:
            result = await chat_json(retry_messages, max_tokens=2500)
        except Exception as e2:
            log.warning("write_implications attempt 2 failed: %s", e2, exc_info=True)
            return {**EMPTY_IMPL, "tldr": "Implication generation failed — see raw facts."}

    # Quality validation: catch hedging / missing falsification fields
    # / generic actions. One retry with critique, then drop bad
    # scenarios rather than persist weak output.
    if result is not None and isinstance(result.get("scenarios"), list):
        issues = _validate_scenarios(result["scenarios"])
        if issues:
            log.info(
                "write_implications: validator caught %d weak scenarios — retrying with critique",
                len(issues),
            )
            retry_result = await _retry_implications_with_critique(
                messages, result, issues
            )
            if retry_result and isinstance(retry_result.get("scenarios"), list):
                post_retry_issues = _validate_scenarios(retry_result["scenarios"])
                if post_retry_issues:
                    bad = {i for i, _ in post_retry_issues}
                    retry_result["scenarios"] = [
                        s for i, s in enumerate(retry_result["scenarios"])
                        if i not in bad
                    ]
                    log.info(
                        "write_implications: dropped %d scenarios still failing post-retry",
                        len(bad),
                    )
                result = retry_result
            else:
                # Retry didn't produce usable output — drop bad scenarios
                # from the original rather than ship them.
                bad = {i for i, _ in issues}
                result["scenarios"] = [
                    s for i, s in enumerate(result["scenarios"]) if i not in bad
                ]
                log.info(
                    "write_implications: retry failed — dropped %d weak scenarios from original",
                    len(bad),
                )

    # Backfill angles[] from scenarios[] for backward-compat readers
    # (daily_brief_flow, preview cards). When the LLM returned the old
    # angles shape (legacy prompt drift), keep what it sent.
    if result is not None and not result.get("angles"):
        result["angles"] = _scenarios_to_angles(result.get("scenarios") or [])
    return result or EMPTY_IMPL
