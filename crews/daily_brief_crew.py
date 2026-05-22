"""Daily brief crew (A, daily not weekly) — 4 agents collaborate on the
250-word morning narrative that lands on the MD's screen at 04:00 GST.

Runs once a day from the scheduler (see flows/daily_brief_flow.py).
Output is stored in daily_briefs.narrative.

Agents:
  1. Researcher    — pulls last 24h of kept signals from Supabase, groups them
  2. PatternFinder — detects surges / inflections / correlations across devs
  3. Writer        — drafts a 250-word MD-voice narrative
  4. FactChecker   — verifies every claim has a source URL in the signals

Why 4 agents instead of 1 big prompt: catches hallucinations (fact_checker),
produces better narrative structure (separate researcher vs writer),
and the pattern step often surfaces things a single-shot LLM misses.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from crewai import Agent, Crew, Process, Task

from tools import sobha_context, supabase_tool as sb
from tools.crewai_llm import nvidia_llm
from tools.observability import observe

log = logging.getLogger(__name__)


def _fetch_recent_signals(hours: int = 24, limit: int = 60) -> list[dict[str, Any]]:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return (
        sb.client()
        .table("signals")
        .select("*")
        .eq("archived", False)
        .gte("last_seen_at", since)
        .order("last_seen_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )


def _compact(sigs: list[dict[str, Any]]) -> str:
    # Signal IDs are required so the Writer's (#N) citations point at
    # real rows the dashboard can drill into. Without them the LLM
    # invents low numbers like (#42) that don't exist (verified
    # 2026-04-26 — brief cited #42-47 when actual IDs were #56-106).
    # URLs are omitted — attribution is rendered separately by the
    # dashboard from daily_briefs.top_signals.
    if not sigs:
        return "(no signals landed in the last 24 hours)"
    lines = []
    for s in sigs:
        sid = s.get("id")
        id_tag = f"#{sid} " if sid else ""
        lines.append(
            f"- {id_tag}dev={s.get('dev_slug') or 'other'} | "
            f"pri={s.get('priority','?')} conf={s.get('confidence','?')} | "
            f"{s.get('headline','—')} — {s.get('dek') or ''}"
        )
    return "\n".join(lines)


_URL_RX = re.compile(r"https?://\S+", re.IGNORECASE)
_BARE_DOMAIN_RX = re.compile(r"\b(?:available at|see|source:|via)\s+\S+\.\S+\b", re.IGNORECASE)


def _strip_urls(text: str) -> str:
    """Strip raw URLs / bare domains. KEEP (#N) signal-id markers — they
    render as clickable ↗ source chips on the dashboard (2026-04-29
    doctrine reversal: previously stripped, now load-bearing for the
    inline-source-link UX). 2026-04-30 audit-round: removed dead
    _CITATION_RX which the comment said was no longer applied."""
    if not text:
        return text
    cleaned = _URL_RX.sub("", text)
    cleaned = _BARE_DOMAIN_RX.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


# 2026-04-30 (audit-round 3): writer keeps inventing cite IDs for the
# market line ("Brent crude is at $103, down 12% (#1)") even when the
# prompt forbids it — those IDs aren't real signals, so they leak as
# dead cite-stubs on the dashboard. Belt-and-suspenders: post-process
# the narrative to drop any (#N) cite that appears inside a sentence
# containing market-line vocabulary.
_MARKET_SENTENCE_TOKENS = (
    "brent", "wti", "gold", "s&p", "s&p 500", "dxy", "nasdaq",
    "us 10y", "us 5y", "us 3m", "treasury yield", "usd/inr",
    "bitcoin", "btc", "eth", "ether", "aed steady", "aed peg",
    "markets:", "crude oil",
)
_SENT_CITE_RX = re.compile(r"\s*\(#\d+\)|\s*#\d+\b")


def _strip_market_line_cites(text: str) -> str:
    if not text:
        return text
    out_parts: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        if any(tok in low for tok in _MARKET_SENTENCE_TOKENS):
            out_parts.append(_SENT_CITE_RX.sub("", sent))
        else:
            out_parts.append(sent)
    cleaned = " ".join(p for p in out_parts if p)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


# 2026-04-30 stakeholder finding: the writer LLM fabricates headlines
# and attributes them to unrelated signal IDs. e.g. wrote "Emaar
# launched 500 units in Downtown Dubai (#245)" when signal #245 was
# actually "Apple Safari security update". The check_task only
# verifies the cite ID exists in the input list, not whether the
# claim matches the signal content. This post-process verifier:
#   1. Walks every (#N) in the narrative
#   2. Fetches that signal's headline+dek from the in-memory map
#   3. Checks token overlap with the writer's sentence
#   4. Drops the entire sentence (and its cite) if overlap is too low
# Net: no more "Apple iOS" being called "Emaar Downtown launch".

_NOISE_TOKENS = {
    "the", "and", "of", "in", "a", "an", "to", "for", "with", "on",
    "at", "by", "as", "is", "are", "was", "were", "from", "into",
    "that", "this", "these", "those", "it", "its", "their", "his",
    "her", "they", "we", "you", "i", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "but", "or", "if",
    "because", "while", "after", "before", "during", "about",
    "also", "too", "any", "all", "some", "no", "not", "than",
    "then", "so", "just", "only", "up", "down", "out", "over",
    "under", "off", "new", "appointed", "launched", "announced",
    "partnership", "director", "ceo", "head", "chief",
}


def _sentence_token_set(text: str) -> set[str]:
    """Lowercase tokens of length>=4, minus stopwords/common verbs."""
    if not text:
        return set()
    raw = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    return {t for t in raw if len(t) >= 4 and t not in _NOISE_TOKENS}


# 2026-04-30 stakeholder finding: writer keeps producing 2-3 director-
# appointment sentences in a row even though the prompt says max 1.
# Programmatic cap: detect appointment sentences and keep only the
# first. Frees the slot for tech/PropTech/regulatory items.
_APPOINTMENT_RX = re.compile(
    r"\b(appoint(?:ed|s|ment)?|named|hired|joins|joined|promoted|"
    r"new (?:director|ceo|head|managing director|president|chief))\b",
    re.IGNORECASE,
)


def _cap_director_appointments(narrative: str, max_keep: int = 1) -> str:
    """Keep at most `max_keep` director-appointment sentences. Drops
    the rest (the LLM ignored the prompt-level diversity mandate)."""
    if not narrative:
        return narrative
    out: list[str] = []
    seen = 0
    for sent in re.split(r"(?<=[.!?])\s+", narrative):
        if _APPOINTMENT_RX.search(sent):
            if seen >= max_keep:
                continue
            seen += 1
        out.append(sent)
    return " ".join(s for s in out if s.strip()).strip()


def _verify_cites_against_signals(
    narrative: str,
    sigs: list[dict[str, Any]],
    min_overlap: int = 2,
) -> str:
    """Drop sentences whose (#N) cite doesn't actually match the cited
    signal's headline+dek. min_overlap = number of meaningful tokens
    that must appear in BOTH the writer's sentence and the signal's
    content. 2 is generous — entity name + a number/community is
    usually enough; total fabrications fail trivially.
    """
    if not narrative or not sigs:
        return narrative
    by_id: dict[int, str] = {}
    for s in sigs:
        sid = s.get("id")
        if sid is None:
            continue
        try:
            by_id[int(sid)] = ((s.get("headline") or "") + " "
                               + (s.get("dek") or ""))
        except Exception:
            continue
    if not by_id:
        return narrative
    out_sentences: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", narrative):
        cite_ids = re.findall(r"\(#(\d+)\)", sent)
        if not cite_ids:
            # No cite — keep (the market-line opener has no cite by design).
            out_sentences.append(sent)
            continue
        sent_tokens = _sentence_token_set(sent)
        # Sentence passes if AT LEAST ONE cited signal has overlap >=
        # min_overlap with the writer's sentence. Multi-cite sentences
        # ("X and Y both did Z (#1)(#2)") only need one match.
        passed = False
        for cid in cite_ids:
            try:
                signal_text = by_id.get(int(cid)) or ""
            except Exception:
                continue
            if not signal_text:
                continue
            sig_tokens = _sentence_token_set(signal_text)
            overlap = len(sent_tokens & sig_tokens)
            if overlap >= min_overlap:
                passed = True
                break
        if passed:
            out_sentences.append(sent)
        # else: drop the sentence entirely — fabrication detected
    cleaned = " ".join(s for s in out_sentences if s.strip())
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _build_crew(
    signals_blob: str,
    signal_count: int,
    market_one_liner: str = "",
) -> Crew:
    # Sobha portfolio context + KPI lens. Threaded into every agent's
    # backstory under the External-Subject / Sobha-Anchor architecture:
    # Sobha is the reference frame, never the subject of a claim.
    sobha_summary = sobha_context.format_for_llm("summary")
    kpi_lens = sobha_context.format_for_llm("kpi_lens")
    persona = sobha_context.PERSONA_PREAMBLE
    market_block = (
        f"\n\nMARKET BACKDROP (for opening line of brief):\n  {market_one_liner}"
        if market_one_liner else ""
    )

    researcher = Agent(
        role="Dubai RE intelligence analyst",
        goal=(
            "Group the last 24h of scored signals by theme: pricing moves, "
            "capital activity, launch cadence, regulatory exposure, off-the-radar. "
            "Call out the one theme that moved the most since yesterday. "
            "Flag any signal that touches Sobha's actual portfolio (named "
            "projects in the context block) as priority."
        ),
        backstory=(
            "15 years covering Gulf developers. Knows every developer's tell. "
            "Starts from the data, never from the narrative.\n\n"
            f"Sobha portfolio context:\n{sobha_summary}"
        ),
        llm=nvidia_llm,
        allow_delegation=False,
    )

    pattern_finder = Agent(
        role="Quantitative trend analyst",
        goal=(
            "Across the grouped themes, detect surges, inflections, or "
            "correlations between developers. Flag anything that looks like "
            "a pattern forming vs isolated noise."
        ),
        backstory=(
            "Looks at any list of events and sees the shape. "
            "Callouts, not commentary."
        ),
        llm=nvidia_llm,
        allow_delegation=False,
    )

    writer = Agent(
        role="MD's chief of staff",
        goal=(
            "Draft a 250-word morning brief for Francis Alfred, MD of Sobha Realty. "
            "EVERY sentence's subject is an EXTERNAL move (a competitor, a "
            "regulator, a capital flow, a macro shift, an infrastructure "
            "announcement). Sobha appears only as the lens — as exposure call "
            "or KPI benchmark. NEVER lead a sentence with Sobha's internal "
            "facts ('Sobha One sold X%', 'Hartland is...') — the MD already "
            "knows that. \n"
            "Preferred shape per claim: external move → mechanism (pricing/"
            "supply/capital/regulation) → KPI lens (where it sits on PSF / "
            "absorption / GP%) → Sobha exposure (which community, project, "
            "or segment is touched, OR explicit 'no direct exposure'). \n"
            "After every concrete claim (name, number, deal, price, date), "
            "append the source signal id in parentheses: '(#42)' — the "
            "dashboard renders these as click-to-drill links. \n"
            "If a needed KPI is missing from the input signals, flag it: "
            "'Need: <specific question>. Delegate to: <web_search | DLD lookup "
            "| Sobha analyst | nobody>.' Never fabricate. \n"
            "Prose only — no bullets, no greeting, no sign-off, no URLs, no "
            "hedge words ('may / could / potentially / monitor'). Write like "
            "the FT Weekend lede."
        ),
        backstory=(
            "Writes for a busy CEO. Every word earns its place.\n\n"
            f"{persona}\n\n"
            f"{sobha_summary}\n\n"
            f"{kpi_lens}"
            f"{market_block}"
        ),
        llm=nvidia_llm,
        allow_delegation=False,
    )

    fact_checker = Agent(
        role="Compliance editor",
        goal=(
            "Verify every concrete claim (name, number, deal, price, date) "
            "in the draft maps to a signal in the input list. If any claim "
            "cannot be sourced, rewrite that sentence to hedge or remove it. "
            "Strip any URLs, links, or domain names that slipped into the prose. "
            "Return the cleaned 250-word brief only — no meta commentary."
        ),
        backstory=(
            "Ex-reuters desk editor. Zero tolerance for unsourced claims."
        ),
        llm=nvidia_llm,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"{signal_count} scored signals landed in the last 24 hours:\n\n"
            f"{signals_blob}\n\n"
            "Group them by theme. Output a numbered list of themes (max 5), "
            "each with a one-line summary and 2-3 signal references from the list."
        ),
        expected_output="Up to 5 numbered themes with supporting signals.",
        agent=researcher,
    )

    pattern_task = Task(
        description=(
            "Review the researcher's themes. Across developers and categories, "
            "label any patterns forming (e.g. 'tier-1 pipeline stall', "
            "'luxury payment-plan compression'). If no pattern, say so explicitly."
        ),
        expected_output="A short bulleted pattern list (max 3 items) or 'no pattern'.",
        agent=pattern_finder,
        context=[research_task],
    )

    write_task = Task(
        description=(
            "Write a 200-word morning narrative for the MD. "
            "POINTERS, NOT ANALYSIS (2026-04-29 stakeholder rule). "
            "Your job is to surface FACTS the MD can see at a glance — "
            "what happened, who did it, with what number. The strategic "
            "read happens later, on demand, behind an Analyze button. "
            "Your narrative must be neutral and attributable.\n\n"
            "OPEN with one short market line — Brent / S&P / gold / DXY / "
            "AED / one Indian-buyer FX — taken VERBATIM from the MARKET "
            "BACKDROP block in your backstory. Do NOT invent market values. "
            "If the backdrop is missing, skip the market line and lead with "
            "the strongest concrete fact.\n"
            "DO NOT append (#N) cites to the market line — market data has "
            "no signal id, and inventing one (e.g. '(#1)') leaks broken cite "
            "chips into the dashboard. The market line is the ONLY sentence "
            "in your brief that must NOT carry a cite token.\n\n"
            "Then pivot to the substance: 6-8 short factual sentences, each "
            "naming a concrete entity + a number or specific move. Plain "
            "prose. No 'Dear MD'.\n\n"
            "DIVERSITY MANDATE (2026-04-30 stakeholder feedback):\n"
            "  Cover DIFFERENT angles across the 4-6 sentences. The MD "
            "  reads this once; redundant items (3 director appointments "
            "  in a row, 3 commodity prices in a row) waste their time.\n"
            "  Aim for one item from each, in order of strongest first:\n"
            "    1. Peer LAUNCH or DEAL (Sobha competitor moving units / acquiring)\n"
            "    2. Capital / market move (sukuk / IPO / EIBOR / FX with RE buyer impact)\n"
            "    3. Tech / AI / PropTech (Anthropic / OpenAI / agentic / BIM / AI-design — these are\n"
            "       interesting awareness items even when not RE-anchored)\n"
            "    4. Regulatory / visa / DLD / RERA / golden-visa-threshold\n"
            "    5. ConTech innovation (3D-print / modular / robotics / fast-build / BIM)\n"
            "    6. Macro / capital-flow (HNWI migration / India LRS / sanctions with RE impact)\n"
            "  HARD CAP: max ONE 'X appointed a new director' or 'leadership "
            "  hiring' sentence per brief. Three 'appointed director' lines "
            "  in a row makes the brief look lazy. If 5 director appointments "
            "  showed up in the input, pick the most material ONE and skip the rest.\n"
            "  HARD CAP: max ONE commodity-price sentence per brief unless "
            "  the move is >5% in a single day.\n"
            "  PREFER variety over completeness — the MD wants a wide-angle "
            "  read, not a category-by-category tally.\n\n"
            "ALLOWED sentence shapes:\n"
            "  - 'Aldar bought the KEZAD portfolio for $177m.'\n"
            "  - 'Modon launched 1,800 units at AED 2,400 PSF in Reem Island.'\n"
            "  - 'Binghatti posted a coming-soon page for a nature-led tower in JVT.'\n"
            "  - 'EIBOR 3M moved from 4.45% to 4.50%.'\n"
            "  - 'Anthropic released Claude 4.7 with a 1M context window.'\n"
            "  - 'Singapore raised the foreign-buyer stamp duty to 60%.'\n\n"
            "BANNED — these belong to the Analyze button, NOT to your narrative:\n"
            "  ✘ 'pressures Sobha pricing'  /  'risks to Sobha'  /  'windows for Sobha'\n"
            "  ✘ 'implies', 'indicates', 'supports', 'threatens', 'supports the case for'\n"
            "  ✘ 'watch this for...' / 'the read is...' / 'this means...'\n"
            "  ✘ Any verb that infers a downstream consequence about Sobha.\n"
            "  ✘ Any 'monitor X', 'consider Y', 'evaluate Z' framing — that's analysis.\n\n"
            "Do not invent numbers or names — only use what the researcher surfaced. "
            "After each concrete claim (a name, number, deal, price, date), "
            "append the source signal id in parentheses: '(#42)'.\n"
            "ANTI-HALLUCINATION (2026-04-30): each sentence must report "
            "ONE signal's facts FAITHFULLY. Do NOT:\n"
            "  ✘ merge two signals into one sentence and stamp one ID\n"
            "  ✘ invent context not in the signal (no 'expanding into Saudi' "
            "    if the signal didn't mention Saudi)\n"
            "  ✘ substitute developer names (if signal says 'Fakhruddin', "
            "    don't write 'Modon')\n"
            "  ✘ invent unit counts, PSF prices, or community names\n"
            "  ✘ paraphrase headlines past recognition\n"
            "If the researcher's line says 'EMAAR appoints director Tariq', "
            "you write 'EMAAR appointed a new director, Tariq (#42).' — NOT "
            "'EMAAR launched 500 units in Downtown Dubai (#42).'\n"
            "A downstream verifier checks token overlap between your sentence "
            "and the cited signal's headline+dek; sentences that fail are "
            "DROPPED from the brief entirely. Hallucinations leave gaps in "
            "your output, not extra content. Stay faithful.\n"
            "CRITICAL: use ONLY the signal IDs that appear at the START of "
            "each line in the researcher's input ('- #N dev=...'). NEVER "
            "invent or guess an ID. If you can't find an ID for a claim, "
            "drop the claim. (These render as clickable source links on the "
            "dashboard — every (#N) becomes a ↗ chip that opens the source.)"
        ),
        expected_output="A 250-word factual brief opening with a market line, then 6-8 attributable fact sentences covering diverse angles (peer launch / capital / tech-AI / regulatory / contech / macro), with inline (#id) citations.",
        agent=writer,
        context=[research_task, pattern_task],
    )

    check_task = Task(
        description=(
            "Read the writer's draft. Compare each factual claim against the "
            "original signals list.\n\n"
            "RULES (absolute):\n"
            "- KEEP every (#N) signal-id marker the writer attached to a claim "
            "  — these render as clickable ↗ source chips on the dashboard "
            "  (2026-04-29 doctrine). If the writer FORGOT to attach an id "
            "  to a verifiable claim, look it up in the signals list and add "
            "  the (#N) marker yourself. Only strip a marker if its id is "
            "  not actually in the input list (i.e. fabricated).\n"
            "- DO NOT introduce hedge words ('may', 'could', 'potentially', "
            "  'monitor', 'might'). If a claim is unsourced, REMOVE the sentence "
            "  rather than soften it.\n"
            "- DO NOT introduce interpretation verbs either ('pressures', "
            "  'implies', 'indicates', 'supports', 'threatens', 'risks to Sobha', "
            "  'windows for', 'opportunities to', 'monitor', 'evaluate', "
            "  'consider'). If the writer slipped any in, rewrite the sentence "
            "  back to the underlying fact only.\n"
            "- DO NOT rewrite in your own voice. Edit-by-deletion + "
            "  interpretation-strip only. Preserve the writer's wording for "
            "  every sentence that passes both the source check and the "
            "  no-interpretation check.\n"
            "- Strip any actual URLs / domain names that slipped in (the "
            "  (#N) markers handle attribution).\n\n"
            "Return the final ~200-word brief with all valid (#N) markers KEPT."
        ),
        expected_output="The final ~200-word brief, factual prose, with every claim carrying a valid (#N) source marker.",
        agent=fact_checker,
        context=[research_task, write_task],
    )

    return Crew(
        agents=[researcher, pattern_finder, writer, fact_checker],
        tasks=[research_task, pattern_task, write_task, check_task],
        process=Process.sequential,
        verbose=False,
        tracing=True,  # streams kickoffs to app.crewai.com (AMP) when authed
    )


@observe(name="daily_brief_crew")
async def run(market_one_liner: str = "") -> dict[str, Any]:
    sigs = _fetch_recent_signals(hours=24)
    crew = _build_crew(_compact(sigs), len(sigs), market_one_liner=market_one_liner)
    result = await asyncio.to_thread(crew.kickoff)
    raw = str(result)
    # 2026-04-30 hallucination guard. Pipeline:
    #   1. _strip_urls — drop raw URLs / bare domains
    #   2. _strip_market_line_cites — strip cites the LLM invented for
    #      the Brent/S&P/etc. opener (no signal IDs there to cite)
    #   3. _verify_cites_against_signals — drop any sentence whose
    #      (#N) cite doesn't actually overlap with the cited signal's
    #      headline+dek. Catches fabricated headlines stamped on real
    #      signal IDs (e.g. "Emaar Downtown launch (#245)" where #245
    #      is "Apple Safari security update").
    cleaned = _strip_urls(raw)
    cleaned = _strip_market_line_cites(cleaned)
    # 2026-04-30 (round B): verifier was nuking entire briefs when
    # the LLM's wording diverged from signal headlines (even when
    # faithful in spirit). Loosen min_overlap 2→1 + add a safety
    # net: if the verifier would drop > 50% of sentences, FALL BACK
    # to the post-strip narrative without verification. Better to
    # ship a slightly-paraphrased brief than an empty one.
    pre_verify = cleaned
    cleaned = _verify_cites_against_signals(cleaned, sigs, min_overlap=1)
    pre_sents = [s for s in re.split(r"(?<=[.!?])\s+", pre_verify) if s.strip()]
    post_sents = [s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if pre_sents and len(post_sents) < max(2, len(pre_sents) // 2):
        log.warning(
            "verifier dropped %d/%d sentences — falling back to pre-verify "
            "narrative to avoid empty brief.",
            len(pre_sents) - len(post_sents), len(pre_sents),
        )
        cleaned = pre_verify
    # Final pass — enforce the diversity mandate the LLM ignores in
    # the prompt: max 1 director-appointment sentence per brief.
    cleaned = _cap_director_appointments(cleaned, max_keep=1)
    return {
        "narrative": cleaned,
        "signal_count": len(sigs),
        "generator_model": nvidia_llm.model,
    }
