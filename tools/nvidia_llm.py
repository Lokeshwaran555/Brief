"""NVIDIA NIM — OpenAI-compatible Llama-3.3 transport.

Same prompts as api/_llm.js in the dashboard; ported to Python + httpx.
Defensive JSON parsing because Llama models through NIM don't always honour
`response_format: json_object`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from settings import settings

log = logging.getLogger(__name__)

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_TIMEOUT = 40.0
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.2


_FAST_ROLES = ("classifier", "fast")


def _chat_provider(role: str = "default") -> tuple[str, str, str]:
    """Resolve (endpoint, api_key, model) for chat.

    Roles:
      - "default"        → main model (LLM_MODEL, e.g. llama-3.3-70b-versatile)
      - "classifier"     → small model (LLM_CLASSIFIER_MODEL, e.g. 8b-instant)
                           Used per raw signal during ingest classification.
      - "fast"           → same small model as "classifier". Used for
                           short format-following calls (brief category
                           synthesis, pointer JSON) where 8B quality is
                           sufficient and the 70B TPM budget should be
                           preserved for the narrative crew.

    Embeddings always stay on NIM (Groq has no embedding model).
    """
    if settings.llm_base_url and settings.llm_api_key and settings.llm_model:
        base = settings.llm_base_url.rstrip("/")
        model = settings.llm_model
        if role in _FAST_ROLES and settings.llm_classifier_model:
            model = settings.llm_classifier_model
        return f"{base}/chat/completions", settings.llm_api_key, model
    if not settings.nvidia_api_key:
        raise RuntimeError(
            "No chat LLM configured. Set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL "
            "(recommended: Groq), or set NVIDIA_API_KEY."
        )
    return ENDPOINT, settings.nvidia_api_key, settings.nvidia_model


INTEL_SYSTEM = """You score unofficial market signals for Francis Alfred, MD of Sobha Realty.

A signal is valuable only if:
  (a) non-public or under-circulated — not in mainstream news yet
  (b) actionable against one of: pricing / capital / launch-timing / risk
  (c) specific — names a developer, tower, community, price, date, or counter-party

You receive ONE raw item (social post, listing, YouTube snippet, broker message, forum post). Score it.

Respond with a SINGLE JSON OBJECT ONLY. No prose. No code fences. Schema:
{
  "keep": true|false,
  "category": "competitor|pricing|regulator|capital|supply|demand|tech|materials|geopolitics|capital_markets|global_prime|capital_flow|other",
  "subcategory": "<see taxonomy below>",
  "decisionTag": "pricing|capital|launch|risk|none",
  "priority": "high|medium|low",
  "confidence": 0.0,
  "headline": "\u226470 chars",
  "dek": "\u2264160 chars \u2014 so-what for Sobha",
  "entities": ["..."],
  "region": "dubai|abu_dhabi|usa|australia|other",
  "country_code": "AE|US|AU|IN|...|null",
  "reason": "\u226480 chars \u2014 why kept or discarded"
}

Category guidance (2026-04-28 stakeholder additions):
  - tech            \u2014 PropTech / ConTech / AI design / digital twin / tokenized RE
  - materials       \u2014 steel rebar, cement, copper, container shipping, MEP lead times
  - geopolitics     \u2014 Israel/Iran/Russia/China/Houthi WHEN they have an RE-impact hook
                      (visa, capital flow, supply chain, sanctions)
  - capital_markets \u2014 sukuk, bond yields, listed RE quotes, MAG-7 spillover, CDS
  - global_prime    \u2014 London / Singapore / NYC / Miami / Monaco prime markets
  - capital_flow    \u2014 population growth, migration, India LRS, China outbound, FDI
The pre-existing six (competitor / pricing / regulator / capital / supply / demand / other) still apply.

Subcategory taxonomy (round 7, 2026-04-30 \u2014 LLM-driven routing).
Pick exactly one. ONE signal, ONE page \u2014 no duplications.

  When category=tech, pick from:
    proptech         \u2014 RE software (Yardi/RealPage/AppFolio/MRI/CoStar),
                       listing platforms (Zillow/Redfin/Compass/Opendoor/
                       Bayut/Property Finder/Dubizzle), broker tech, smart
                       home / IoT, tokenized RE, blockchain RE.
    contech          \u2014 build methodology innovation: prefab, modular,
                       3D-print, robotic mason, BIM, digital twin (in
                       construction context), AI-for-design, generative
                       design, Procore, Autodesk Forge, Matterport,
                       fast-build records, mass timber, low-carbon
                       concrete, geopolymer.
    frontier_ai      \u2014 Anthropic/Claude/OpenAI/GPT/DeepMind/Mistral/
                       Perplexity/xAI/Hugging Face moves, model launches,
                       major AI funding rounds, AI-research breakthroughs,
                       agentic AI \u2014 regardless of direct RE bearing.

  When category=materials, pick from:
    materials_cost        \u2014 commodity prices (steel/cement/copper/
                            shipping/diesel) + supply chain disruption.
    materials_innovation  \u2014 new building materials (mass timber,
                            geopolymer, self-healing concrete, recycled
                            aggregate) \u2014 these can ALSO be tagged contech
                            when methodology is the angle.

  When category=regulator, pick from:
    regulatory_visa       \u2014 Golden Visa / property residency /
                            visa thresholds / freehold rules.
    regulatory_dev        \u2014 RERA/DLD/Trakheesi rule changes,
                            developer compliance, escrow.
    regulatory_macro      \u2014 central-bank rate changes, mortgage caps.

  When category=competitor, pick from:
    competitor_launch     \u2014 a tracked developer launches a project.
    competitor_ma         \u2014 M&A / acquisition / partnership.
    competitor_capital    \u2014 sukuk, IPO, capital raise, investor day.
    competitor_talent     \u2014 senior leadership move at a peer.
    competitor_pricing    \u2014 commission/payment-plan/discount move.

  When category=capital_markets:
    capital_sukuk         \u2014 sukuk yields, bond spreads, dev-issued debt.
    capital_listed        \u2014 listed RE equity moves (Emaar, Aldar, Damac).
    capital_fx            \u2014 FX moves with RE-buyer impact (USD/INR, etc.).
    capital_rates         \u2014 EIBOR, US Treasury, mortgage rate moves.

  When category=geopolitics:
    geopolitics_re_impact \u2014 sanctions / Red Sea / China outbound / war
                            with RE-buyer or supply-chain impact.

  When category=global_prime:
    hpi_prime             \u2014 London/Singapore/NYC/Miami/Monaco luxury HPI.

  When category=capital_flow:
    hnwi_migration        \u2014 Henley reports, India LRS, UAE NCS, Golden
                            Visa migration data.
    capital_flow_other    \u2014 population, demographics, FDI not covered above.

  Other / fallback values:
    lifestyle             \u2014 watches, yachts, supercars, Sotheby's
                            auctions \u2014 HNW-spend proxy.
    sales_intel           \u2014 demand-side: rental yields, secondary listings,
                            buyer-nationality data \u2014 when not tagged elsewhere.
    other                 \u2014 use only when nothing else fits.

Routing principle: ONE signal lands on ONE themed page. The MD reviews
each page expecting unique content. If a story plausibly fits two
subcategories, pick the one where it's most actionable for Sobha.

PAGE-ROUTING TABLE (2026-05-08 stakeholder directive — be aggressive
about routing, not selective about keeping). The MD has 50 scouts
producing 500+ signals/day; the failure mode is empty tabs, not
overflow. Default to keep=true for anything that could plausibly
move a Sobha decision, then route precisely:

  ┌─ MD SCAN tab (region-bucketed: dubai / abu_dhabi / usa / australia / global)
  │   Anything that affects real estate or HNW / luxury / institutional
  │   capital flow in the named region. Includes:
  │     - developer launches, partnerships, M&A, capital raises
  │     - DLD / RERA / regulatory moves
  │     - residential / commercial / hospitality / branded residences
  │     - HNW migration, Golden Visa, foreign-buyer demand
  │     - infrastructure (metro, airport, smart city) that shifts land value
  │     - geopolitics with direct RE-buyer / capital-flow impact
  │     - distress, restructuring, bankruptcy in the dev set
  │     - sales velocity / absorption / secondary market
  │     - broker / talent moves at peers
  │   Region tag drives which sub-tab the signal lands on.
  │
  ├─ MARKETS & CAPITAL tab (category=capital_markets, capital, regulator
  │   when macro, geopolitics_re_impact, global_prime, capital_flow,
  │   capital_*, regulatory_macro, hpi_prime, hnwi_migration)
  │   Anything that prices money or moves capital — rates, sukuk, bonds,
  │   FX, listed RE equities, MAG-7 spillover, central-bank moves,
  │   sovereign-wealth deployments, REIT IPOs.
  │
  ├─ CONSTRUCTION ECONOMICS tab (category=materials, supply, contech
  │   subcategories materials_cost, materials_innovation)
  │   Material prices, shipping, MEP lead times, supply-chain
  │   disruption, prefab/modular methodology.
  │
  └─ TECH & AI tab (category=tech, subcategories proptech, contech,
      frontier_ai)
      PropTech, ConTech, AI design, BIM, digital twin, tokenized RE,
      AI model launches, AI funding rounds.

PRIMARY CATEGORY → PAGE MAP (use this to set the `category` field
when subcategory taxonomy doesn't fit cleanly):
  - Real-estate-in-region story  → category=competitor|pricing|supply|
                                    demand|regulator (MD Scan tab)
  - Capital / money / rates story → category=capital_markets|capital|
                                    capital_flow|global_prime
                                    (Markets & Capital tab)
  - Build-cost / shipping story  → category=materials|supply
                                    (Construction tab)
  - Software / AI / model story  → category=tech (Tech & AI tab)

KEEP THRESHOLD (loosened 2026-05-08): keep=true if the signal names
ANY of: a developer (any region) · a city / community · a price / unit
count / payment plan · a regulatory body · a sukuk / bond / IPO · a
material commodity · a PropTech / ConTech / AI tool · a HNW migration
data point. Mainstream / front-page coverage is fine to keep at
priority=medium when it confirms a trend.

NO EDITORIAL JUDGEMENT (2026-05-07 stakeholder directive).
The classifier reports facts, never opinions. The headline + dek must
describe WHAT happened — who, what, where, when, how much. They must
NOT contain:
  - Recommendations ("Sobha should...", "consider matching this", "watch
    for impact")
  - Speculation ("could mean", "may signal", "likely indicates")
  - Editorial framing ("worryingly", "promisingly", "notably")
  - Strategic positioning ("a threat", "an opportunity", "a wake-up call")
The "reason" field is internal and may be terse, but headline/dek are
external — keep them as a wire-service reporter would write them.

Region tagging rules (drives the MD Scan dashboard's region tabs):
  - "dubai"      \u2014 anything in/about Dubai (Hartland, Reem Island in Dubai
                       context, Marina, Downtown, MBR City, Meydan, JLT, Palm,
                       Emaar, DAMAC, Nakheel, Binghatti, Azizi, Dubai Holding).
  - "abu_dhabi"  \u2014 Abu Dhabi only (Saadiyat, Al Maryah, Yas, Reem Island
                       in AD context, Aldar, Mubadala, IHC, ADQ, ADGM).
                       Do NOT default UAE \u2192 dubai when the signal is AD-specific.
  - "usa"        \u2014 anywhere in the United States (Texas, Austin, Dallas,
                       Houston, NYC, FL, REIT activity, US multifamily, SEC).
  - "australia"  \u2014 anywhere in Australia (Brisbane, Sydney, Melbourne, AU
                       developers, Olympic infrastructure, ASIC, ABS).
  - "other"      \u2014 every other country. Use country_code to identify it
                       (PL for Poland, IN for India, VN for Vietnam, ...). If
                       the signal is genuinely cross-border, pick the country
                       most central to the move and tag region=other.
  - country_code is ISO 3166-1 alpha-2. Null only when the signal is
    genuinely region-agnostic (rare).

Keep / discard rules (v3, loosened 2026-05-08 — see KEEP THRESHOLD
above. Default to keep=true if in doubt.):
  - Mainstream Khaleej Times / Gulf News / The National coverage: KEEP
    at priority=medium when it carries a concrete fact (price, unit
    count, dev name, regulatory move). Was previously discarded; that
    was over-pruning the corpus. Only discard when the headline is
    pure opinion or recycled stock content with no new fact.
  - Routine Bayut / Dubizzle / Property Finder resale listings: keep=false.
    Only keep listings when they cluster into a pattern: a developer
    suddenly posts 20+ new units in one community, a brand-new project
    appears on the listings side before the dev announced it, or a
    single listing carries unusual commission / payment-plan terms.
  - LinkedIn company-page posts from tier-1 developers that announce
    launches, partnerships, pipeline or capital moves → high or medium.
  - Old content leaking through: if the body implies the event
    happened months ago (e.g. 'completed in 2024', 'announced last
    year'), keep=false or low priority at best.
  - Personal / lifestyle posts (watch collections, interviews without
    numbers, generic cheerleading): keep=false.
  - Broker commission bumps, pre-launch EOI hints, leadership moves
    at peers, community-level listing deltas > 3% WoW → high.

Priority heuristic:
  high   = directly actionable this week (pricing / launch / partnership)
  medium = interesting but needs follow-up work before it's decidable
  low    = noise-adjacent, worth seeing once but not re-surfacing

Never invent facts. If ambiguous, low confidence."""


def _parse_retry_after(resp: httpx.Response) -> float:
    """Extract retry delay from a 429 response. Groq returns the hint
    inside the JSON error body ('Please try again in 12.05s'); falls
    back to Retry-After header, then a 30s safe default.

    Capped at 60s so a stuck pipeline doesn't block forever.
    """
    try:
        body = resp.json()
        msg = (body.get("error") or {}).get("message", "")
        m = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
        if m:
            return min(float(m.group(1)) + 1.0, 60.0)
    except Exception:
        pass
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return min(float(ra), 60.0)
        except ValueError:
            pass
    return 30.0


async def _call(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    want_json: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    role: str = "default",
    _retry_count: int = 0,
) -> str:
    chat_endpoint, chat_key, chat_model = _chat_provider(role)
    body: dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {chat_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Allow long enough for a 429 + backoff + retry to fit inside one
    # call. The default 40s is fine for one shot; 110s lets us absorb
    # one full 60s backoff plus a real call.
    import asyncio as _asyncio
    eff_timeout = timeout if _retry_count else max(timeout, 110.0)
    async with httpx.AsyncClient(timeout=eff_timeout) as client:
        resp = await client.post(chat_endpoint, json=body, headers=headers)
        # Some providers 400 on response_format — retry without it.
        if resp.status_code == 400 and want_json:
            body.pop("response_format", None)
            resp = await client.post(chat_endpoint, json=body, headers=headers)
        # 429 → parse hint, sleep, retry once. Without this, free-tier
        # TPM bursts (CEO scan after brief, etc.) silently produce
        # empty results that the dashboard can't distinguish from
        # genuine 'no strategic content'.
        if resp.status_code == 429 and _retry_count < 1:
            wait_s = _parse_retry_after(resp)
            log.info(
                "rate limit hit (model=%s role=%s) — waiting %.1fs then retrying once",
                chat_model, role, wait_s,
            )
            await _asyncio.sleep(wait_s)
            return await _call(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                want_json=want_json,
                timeout=timeout,
                role=role,
                _retry_count=_retry_count + 1,
            )
        resp.raise_for_status()
        data = resp.json()

    # Reasoning models (gpt-oss-120b/20b on Groq, OpenAI o-series) emit
    # their final answer in `content` AND a CoT in `reasoning_content`.
    # On complex prompts, gpt-oss occasionally consumes its budget in
    # the reasoning channel and returns content="" — without this
    # fallback, downstream JSON parsing fails ("Implication generation
    # failed — see raw facts" surfaces in the dashboard). When content
    # is empty, treat reasoning_content as the answer of last resort.
    try:
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("reasoning_content") or "").strip()
        if not content:
            raise RuntimeError(f"empty content + empty reasoning_content: {str(msg)[:300]}")
        return content
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"LLM empty response: {str(data)[:300]}") from e


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s


async def chat_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    role: str = "default",
) -> dict[str, Any]:
    raw = await _call(messages, max_tokens=max_tokens, want_json=True, role=role)
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"LLM returned non-JSON. Raw: {cleaned[:300]}")


async def embed(text: str) -> list[float] | None:
    """Return a single embedding vector for `text`, or None on failure.

    Uses NVIDIA NIM's nv-embedqa-e5-v5 (1024 dims). Falling back to None
    lets callers degrade to title-hash dedup so a flaky embed call
    never blocks signal ingest.
    """
    if not text or not text.strip():
        return None
    if not settings.nvidia_api_key:
        # No NIM key → embeddings disabled. Caller falls back to
        # title-hash dedup, which is already the documented degradation.
        return None
    body = {
        "input": [text[:1500]],
        "model": settings.nvidia_embed_model,
        "input_type": "passage",
        "encoding_format": "float",
        "truncate": "END",
    }
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(EMBED_ENDPOINT, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        vec = data["data"][0]["embedding"]
        if not isinstance(vec, list) or len(vec) != settings.nvidia_embed_dims:
            log.warning("embed: unexpected dims=%s", len(vec) if isinstance(vec, list) else type(vec))
            return None
        return vec
    except Exception as e:
        log.warning("embed failed (%s) — falling back to title-hash dedup", e)
        return None


async def classify_intel(signal: dict[str, Any]) -> dict[str, Any]:
    """Score one raw signal. Returns {keep, category, priority, ...}.

    Routed through the classifier-tier model (smaller, cheaper) when
    LLM_CLASSIFIER_MODEL is set. Falls back to the main model otherwise.

    Sobha portfolio summary is appended to the system prompt so the
    classifier can rank by Sobha-relevance, not just generic priority.
    A signal touching Sobha's actual communities/projects gets boosted
    even when it's a "small" market move.
    """
    # Lazy import to keep this module importable without sobha_context
    # in tests / one-shot scripts. The summary call is cached (1h TTL)
    # so cost is one DB read per ingest run, not per classifier call.
    try:
        from tools import sobha_context
        sobha_summary = sobha_context.format_for_llm("summary")
    except Exception:
        sobha_summary = ""

    system_with_context = INTEL_SYSTEM
    if sobha_summary:
        system_with_context = (
            INTEL_SYSTEM
            + "\n\n────────────────────────────────────────────\n"
            + "SOBHA PORTFOLIO CONTEXT (use to rank relevance):\n"
            + sobha_summary
            + "\n────────────────────────────────────────────\n"
            + "When the signal touches a Sobha community/project named "
            + "above (or a competitor in those sub-markets), bias toward keep=true "
            + "and priority=high. When the signal is in a region where Sobha "
            + "has no live product, priority caps at medium unless it's a major "
            + "regulatory or capital-flow move."
        )

    return await chat_json(
        [
            {"role": "system", "content": system_with_context},
            {"role": "user", "content": f"Raw signal:\n{json.dumps(signal, indent=2)}"},
        ],
        max_tokens=800,
        role="classifier",
    )
