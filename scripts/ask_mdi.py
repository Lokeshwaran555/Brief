#!/usr/bin/env python3
"""ask_mdi.py — Local POC of a Sobha-tuned MDI assistant (RAG over Supabase).

This is a Cursor-style domain assistant: ask plain-English questions about
the Dubai/AD/USA/AU real-estate intelligence stack, get CEO-altitude
answers grounded in the actual signals + events + briefs we've ingested.

Uses the existing infrastructure (no new accounts):
  - NVIDIA NIM nv-embedqa-e5-v5  (1024-dim embeddings)
  - Supabase pgvector             (top-K similarity search)
  - Groq llama-3.3-70b-versatile  (synthesis)
  - Groq llama-3.1-8b-instant     (cheap fallback if 70B is rate-limited)

Run locally:
  cd ~/projects/sobha-mdi-agents
  source .venv/bin/activate                       # if you have one
  pip install httpx supabase python-dotenv         # one-time
  python scripts/ask_mdi.py "what's hot in abu dhabi this week?"

Env vars expected (mirrors Railway):
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  NVIDIA_API_KEY                                   # for embeddings
  LLM_BASE_URL=https://api.groq.com/openai/v1
  LLM_API_KEY=<groq key>
  LLM_MODEL=llama-3.3-70b-versatile

If LLM_API_KEY isn't set the script prints retrieved context only — still
useful for verifying the retrieval layer.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime, timedelta, timezone

try:
    import httpx
except ImportError:
    print("missing dep: pip install httpx", file=sys.stderr)
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("missing dep: pip install supabase", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


# ─── Persona prompt ──────────────────────────────────────────────────
MDI_SYSTEM = """You are MDI Scanner — a strategic intelligence assistant for
Francis Alfred, Managing Director of Sobha Realty (Dubai luxury developer
with footprint in Abu Dhabi, Texas, Brisbane, plus emerging-market scanning).

You answer questions about competitor moves, market trends, and strategic
opportunities by reasoning over the retrieval context provided.

VOICE
- CEO altitude. Strategic-first. Quantify when possible.
- Lead with the so-what. No throat-clearing, no hedging ("may", "could",
  "potentially", "might"). Commit to a read or say you don't have enough
  data — never both.
- Reference Sobha projects by name when relevant: Hartland, Hartland II,
  One, Reserve, SeaHaven, Skyvue, Verde, Orbis, Elwood, Solis, Downtown.
- Name competitors specifically: Emaar, DAMAC, Aldar, Nakheel, Binghatti,
  Mubadala, IHC, Modon, Mirvac, Stockland, Camden, Mid-America, etc.

WHAT YOU HAVE
The user's question, plus a retrieval pack of:
  - Recent SIGNALS (raw intel — headline, dek, dev, region, priority)
  - INVESTIGATED EVENTS (deep research — facts: units, PSF, payment plan,
    partners; scenarios: opportunity / risk / watch reads)
  - DAILY BRIEFS (recent narrative summaries)
  - CROSS-SIGNAL PATTERNS (synthesised observations — surge, cluster, etc.)

WHAT YOU DON'T HAVE
- Sobha's internal MIS (sales, inventory, pricing). Note 'MIS not wired'
  if the question requires it.
- Real-time market quotes or anything outside the retrieval pack.

OUTPUT
2-3 short paragraphs. No bullet lists unless the question is comparative.
Cite specific signal/event ids in [brackets] when making concrete claims
so the user can drill in.
"""


# ─── Supabase client ─────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("[ask_mdi] missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY env", file=sys.stderr)
    sys.exit(1)

sb = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─── NVIDIA NIM embedding ────────────────────────────────────────────
NIM_API_KEY = os.environ.get("NVIDIA_API_KEY")
NIM_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NIM_EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")


def embed(text: str) -> list[float] | None:
    if not NIM_API_KEY:
        return None
    body = {
        "input": [text[:1500]],
        "model": NIM_EMBED_MODEL,
        "input_type": "query",  # 'query' for retrieval, not 'passage'
        "encoding_format": "float",
        "truncate": "END",
    }
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    try:
        r = httpx.post(NIM_EMBED_URL, json=body, headers=headers, timeout=20.0)
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
        if isinstance(vec, list) and len(vec) == 1024:
            return vec
    except Exception as e:
        print(f"[ask_mdi] embed failed: {e}", file=sys.stderr)
    return None


# ─── Retrieval ───────────────────────────────────────────────────────
def retrieve_signals(query_vec: list[float] | None, *, limit: int = 8, days: int = 14) -> list[dict]:
    """Top-K signals by pgvector cosine similarity (when embedding works),
    else most-recent fallback. Filters to last `days` days.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if query_vec is None:
        rows = (
            sb.table("signals")
            .select("id,headline,dek,category,decision_tag,priority,dev_slug,region,country_code,source_published_at,last_seen_at,urls")
            .eq("archived", False)
            .gte("last_seen_at", since)
            .order("last_seen_at", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        return rows
    # Try pgvector match_signal RPC first. Falls through to recency
    # whenever it errors OR returns too few hits — many signals don't
    # have embeddings populated yet, so empty-RPC is just as common
    # as an exception.
    try:
        rows = sb.rpc(
            "match_signal",
            {"query_embedding": query_vec, "match_threshold": 0.3, "match_count": limit},
        ).execute().data or []
        ids = [r["id"] for r in rows]
        if len(ids) >= 2:
            full = (
                sb.table("signals")
                .select("id,headline,dek,category,decision_tag,priority,dev_slug,region,country_code,source_published_at,last_seen_at,urls")
                .in_("id", ids)
                .execute()
                .data
                or []
            )
            if full:
                return full
        # RPC empty or returned only one match — augment with recent
        # signals so the LLM has material to reason against.
        print(f"[ask_mdi] vector match returned {len(ids)} — augmenting with recency", file=sys.stderr)
    except Exception as e:
        print(f"[ask_mdi] vector retrieval failed ({e}) — falling back to recency", file=sys.stderr)
    return retrieve_signals(None, limit=limit, days=days)


def retrieve_events(signal_ids: list[int], *, limit: int = 5) -> list[dict]:
    if not signal_ids:
        return []
    try:
        rows = (
            sb.table("market_events")
            .select(
                "id,seed_signal_id,event_type,project_name,location,event_date,status,headline,"
                "event_facts(facts,unresolved),event_implications(scenarios,tldr,pointers)"
            )
            .in_("seed_signal_id", signal_ids)
            .limit(limit)
            .execute()
            .data
            or []
        )
        return rows
    except Exception:
        return []


def retrieve_latest_brief() -> dict | None:
    try:
        rows = (
            sb.table("daily_briefs")
            .select("day,narrative,pointers,signal_count")
            .order("day", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except Exception:
        return None


# ─── Context builder ─────────────────────────────────────────────────
def build_context(question: str) -> tuple[str, dict]:
    qvec = embed(question)
    signals = retrieve_signals(qvec, limit=10)
    sig_ids = [int(s["id"]) for s in signals if s.get("id")]
    events = retrieve_events(sig_ids, limit=5)
    brief = retrieve_latest_brief()

    parts: list[str] = []
    if brief:
        nar = (brief.get("narrative") or "")[:600]
        parts.append(f"=== LATEST DAILY BRIEF ({brief.get('day')}) ===\n{nar}\n")
    if signals:
        parts.append("=== RECENT SIGNALS ===")
        for s in signals[:10]:
            parts.append(
                f"[id={s['id']}] {s.get('priority','low').upper()} · {s.get('region','-')} · "
                f"{s.get('dev_slug','-')} · {s.get('decision_tag','-')}"
            )
            parts.append(f"  {s.get('headline','')}")
            if s.get('dek'):
                parts.append(f"  └ {s['dek'][:200]}")
        parts.append("")
    if events:
        parts.append("=== INVESTIGATED EVENTS ===")
        for e in events:
            ef = e.get("event_facts") or {}
            if isinstance(ef, list):
                ef = ef[0] if ef else {}
            ei = e.get("event_implications") or {}
            if isinstance(ei, list):
                ei = ei[0] if ei else {}
            facts = ef.get("facts") or {}
            scenarios = ei.get("scenarios") or []
            parts.append(
                f"[event_id={e['id']}] {e.get('event_type','')} · {e.get('project_name','')} · {e.get('location','')}"
            )
            if facts:
                fact_pairs = [f"{k}={v}" for k, v in facts.items() if v not in (None, "", [])][:6]
                if fact_pairs:
                    parts.append(f"  facts: {', '.join(fact_pairs)}")
            if scenarios:
                for sc in scenarios[:2]:
                    parts.append(
                        f"  scenario [{sc.get('stance','watch')}]: {(sc.get('claim') or '')[:160]}"
                    )
        parts.append("")

    context_text = "\n".join(parts).strip()
    return context_text, {
        "signal_count": len(signals),
        "event_count": len(events),
        "brief_present": bool(brief),
        "embedding_used": qvec is not None,
    }


# ─── Groq inference ──────────────────────────────────────────────────
LLM_BASE_URL = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL") or "llama-3.3-70b-versatile"


def ask_llm(question: str, context: str) -> str:
    if not (LLM_BASE_URL and LLM_API_KEY):
        return "[LLM_API_KEY not set — context-only mode below]"
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": MDI_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nRETRIEVAL CONTEXT:\n{context or '(empty — no recent signals matched)'}",
            },
        ],
        "max_tokens": 900,
        "temperature": 0.3,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    try:
        r = httpx.post(f"{LLM_BASE_URL}/chat/completions", json=body, headers=headers, timeout=60.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM call failed: {e}]"


# ─── CLI ─────────────────────────────────────────────────────────────
def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/ask_mdi.py 'your question'", file=sys.stderr)
        return 1
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        return 1

    print("=" * 72)
    print(f"MDI · {question}")
    print("=" * 72)

    context, meta = build_context(question)
    print(f"\n[retrieval] signals={meta['signal_count']}  events={meta['event_count']}  "
          f"brief={meta['brief_present']}  embed={meta['embedding_used']}")
    print(f"[model] {LLM_MODEL or '(no LLM)'}\n")

    answer = ask_llm(question, context)
    print(textwrap.fill(answer, width=88, replace_whitespace=False, drop_whitespace=False))
    print("\n" + "─" * 72)
    print("[retrieval context used]")
    print(textwrap.indent((context or "(empty)")[:2000], "  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
