"""Deep-dive crew (B) — 3 agents collaborate on a per-developer memo.

Triggered on-demand via POST /deep-dive/{dev_slug}. Pulls every signal we
have on the developer from Supabase, runs a 3-agent chain, returns a
300-word memo for the MD.

Agents:
  1. Researcher  — pulls & summarises signals, public filings, news mentions
  2. PatternFinder — extracts pricing / capital / launch / risk patterns
  3. Writer     — produces a 300-word MD-voice memo

Input:  dev_slug (one of emaar|damac|aldar|nakheel|modon|dubaih|binghatti|azizi|ellington|omniyat|deyaar)
Output: {dev_slug, narrative, kpis, signal_count}
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from crewai import Agent, Crew, Process, Task

from tools import sobha_context, supabase_tool as sb
from tools.crewai_llm import nvidia_llm
from tools.observability import observe

log = logging.getLogger(__name__)


def _fetch_dev_signals(dev_slug: str, limit: int = 50) -> list[dict[str, Any]]:
    return (
        sb.client()
        .table("signals")
        .select("*")
        .eq("dev_slug", dev_slug)
        .eq("archived", False)
        .order("last_seen_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )


def _compact(sigs: list[dict[str, Any]]) -> str:
    if not sigs:
        return "(no signals in the store yet)"
    lines = []
    for s in sigs[:30]:
        lines.append(
            f"- [{s.get('priority','?')}/{s.get('confidence','?')}] "
            f"{s.get('headline','—')} — {s.get('dek') or ''} "
            f"(tag: {s.get('decision_tag') or 'none'}, "
            f"seen: {s.get('source_count', 1)}× since {s.get('first_seen_at','?')})"
        )
    return "\n".join(lines)


def _build_crew(dev_slug: str, signals_blob: str) -> Crew:
    # Sobha overlap with this competitor — names exact regions /
    # communities / adjacent projects so the writer can ground every
    # implication in real exposure, not guess.
    overlap_text = sobha_context.format_overlap_for_llm(dev_slug)

    researcher = Agent(
        role=f"Dubai RE intelligence analyst, {dev_slug} beat",
        goal=(
            f"Pull together every scored signal we have on {dev_slug} and "
            "surface the three strongest threads (pricing moves, capital "
            "activity, launch cadence, leadership changes, regulatory exposure)."
        ),
        backstory=(
            "15 years covering Gulf developers. Reads every IR deck, "
            "every DFM/ADX filing, every broker WhatsApp rumor. "
            "Distills noise into three strands."
        ),
        llm=nvidia_llm,
        verbose=False,
        allow_delegation=False,
    )

    pattern_finder = Agent(
        role="Quantitative pattern analyst",
        goal=(
            "Turn the researcher's threads into explicit patterns: "
            "directional moves, surges vs baseline, correlations between "
            "signal types. Name the pattern in one short phrase each."
        ),
        backstory=(
            "Ex-hedge-fund analyst. Looks at any list of events and sees "
            "the shape — accumulation, distribution, inflection, or noise."
        ),
        llm=nvidia_llm,
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="MD's chief of staff",
        goal=(
            "Write a 300-word memo to Francis Alfred (MD of Sobha Realty) "
            f"about {dev_slug}. Open with the single most important pattern. "
            "Plain prose, no bullets, no hedging. Use the researcher's "
            "threads and the pattern-finder's labels. Anchor every "
            "implication to a specific Sobha asset / community / segment "
            "named in the overlap context — not generic 'Sobha pricing'."
        ),
        backstory=(
            "Writes for a busy CEO. Narrative over numbers. Never pads.\n\n"
            f"Sobha competitive overlap with {dev_slug}:\n{overlap_text}"
        ),
        llm=nvidia_llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"You are briefing the MD on {dev_slug}. Here are the signals "
            "we have in the store on them right now:\n\n"
            f"{signals_blob}\n\n"
            "Return a numbered list of the three most important threads. "
            "For each thread: one-line description + 1-2 supporting signals. "
            "If signals are sparse, say so explicitly and lean on what we do know."
        ),
        expected_output="A numbered list of 3 threads, each with supporting signals.",
        agent=researcher,
    )

    pattern_task = Task(
        description=(
            "Take the researcher's 3 threads. For each, label the pattern "
            "in 1-4 words (e.g. 'price accumulation', 'capital tightening', "
            "'pipeline stall', 'leadership flux'). Return a table: "
            "Thread # | Pattern label | Signal strength (weak/medium/strong)."
        ),
        expected_output="A 3-row table with Thread, Pattern, Strength.",
        agent=pattern_finder,
        context=[research_task],
    )

    write_task = Task(
        description=(
            f"Write the 300-word memo to Francis Alfred about {dev_slug}. "
            "Lead with the strongest pattern from the pattern-finder's table. "
            "Second paragraph: second pattern + implication for Sobha — "
            "name a specific Sobha asset/community/segment from the overlap "
            "context in your backstory. "
            "Third paragraph: what Sobha should watch this week, again "
            "anchored to the overlap context. "
            "Prose only — no bullets, no headings, no 'Dear MD'. "
            "If the overlap context shows no direct head-to-head, say so "
            "explicitly and frame the memo as 'watch-list', not response."
        ),
        expected_output="A single 300-word memo in plain prose.",
        agent=writer,
        context=[research_task, pattern_task],
    )

    return Crew(
        agents=[researcher, pattern_finder, writer],
        tasks=[research_task, pattern_task, write_task],
        process=Process.sequential,
        verbose=False,
        tracing=True,  # streams kickoffs to app.crewai.com (AMP) when authed
    )


@observe(name="deep_dive_crew")
async def run(dev_slug: str) -> dict[str, Any]:
    sigs = _fetch_dev_signals(dev_slug)
    signals_blob = _compact(sigs)
    crew = _build_crew(dev_slug, signals_blob)

    # CrewAI kickoff is sync; run in a worker thread so the FastAPI loop
    # stays responsive.
    result = await asyncio.to_thread(crew.kickoff)
    narrative = str(result)

    return {
        "dev_slug": dev_slug,
        "signal_count": len(sigs),
        "narrative": narrative,
        "generator_model": nvidia_llm.model,
    }
