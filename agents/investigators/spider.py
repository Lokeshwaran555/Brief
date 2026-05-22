"""Investigation spider — deterministic data-gathering stage.

Takes resolved entities (project, developer, location) and fans out
across every source we have, returning a flat list of evidence items:
  [{source, source_url, title, excerpt, raw_json}, ...]

No LLMs here — this is pure I/O. The Crew consumes the evidence list
in the next stage. Keeping gather and reasoning separate makes the
investigation fast, cheap, and debuggable.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from tools import apify_youtube, supabase_tool as sb, tavily_search, yt_transcript

log = logging.getLogger(__name__)


def _q(entities: dict[str, Any]) -> str:
    """Compose a natural-language search query from the entity dict.

    Multi-word entities are quoted so Tavily treats them as phrases —
    otherwise a query like `Emaar Dubai` matches any Dubai property
    article that happens to mention Emaar in passing, flooding the
    evidence pack with regional noise (Egypt cities, workforce PR, etc.)
    """
    parts: list[str] = []
    for k in ("project", "developer", "location"):
        v = entities.get(k)
        if not v:
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(f'"{s}"' if " " in s else s)
    composed = " ".join(parts).strip()
    if composed:
        return composed
    # Fallbacks when LLM entity resolution failed — use whatever we have.
    return (entities.get("headline") or entities.get("developer_slug") or "").strip()


# Tavily scores are 0-1 relevance. Below this, results are typically
# tangential matches — kept out of the evidence pack to keep fact
# extraction honest. Empirically 0.3 keeps real on-topic news (~0.4-0.8)
# and drops the regional-noise floor (~0.1-0.25).
TAVILY_MIN_SCORE = 0.3

# Hard age floor for evidence. Investigations occasionally need
# older context (a 6-month-old land deal that's now closing) so we
# keep a longer window than the scout, but a year-old LinkedIn post
# being treated as fresh evidence is the failure mode this kills.
MAX_EVIDENCE_AGE_DAYS = 90


def _published_within_age(published: Any, max_days: int) -> bool:
    if not published:
        return True  # unknown date passes — classifier handles age decay
    try:
        s = str(published).rstrip("Z")
        dt = datetime.fromisoformat(s)
    except Exception:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt) <= timedelta(days=max_days)


async def _tavily_block(query: str) -> list[dict[str, Any]]:
    if not query:
        return []
    general, news = await asyncio.gather(
        tavily_search.search(query, max_results=10, topic="general", search_depth="advanced"),
        tavily_search.search(query, max_results=10, topic="news", days=30),
        return_exceptions=True,
    )
    out: list[dict[str, Any]] = []
    dropped_low_score = 0
    dropped_too_old = 0
    for r in (general, news):
        if isinstance(r, Exception):
            log.warning("[spider:tavily] batch raised: %s", r)
            continue
        for item in r:
            score = item.get("score")
            if isinstance(score, (int, float)) and score < TAVILY_MIN_SCORE:
                dropped_low_score += 1
                continue
            published = item.get("published_date")
            if not _published_within_age(published, MAX_EVIDENCE_AGE_DAYS):
                dropped_too_old += 1
                continue
            out.append(
                {
                    "source": "tavily",
                    "source_url": item.get("url"),
                    "title": (item.get("title") or "")[:300],
                    "excerpt": (item.get("content") or "")[:800],
                    "source_published_at": published,
                    "raw_json": {
                        "score": score,
                        "published_date": published,
                    },
                }
            )
    if dropped_low_score or dropped_too_old:
        log.info(
            "[spider:tavily] dropped %d low-relevance, %d too-old (>%dd)",
            dropped_low_score, dropped_too_old, MAX_EVIDENCE_AGE_DAYS,
        )
    return out


async def _yt_block(query: str) -> list[dict[str, Any]]:
    """YT search → attach transcripts for the top 6 hits.

    Apify's YT search returns titles + minimal metadata only, so the LLM
    downstream gets near-zero text. Fetching transcripts for the top
    hits (most-recent / highest-view) is free via youtube-transcript-api
    and turns YT from a title-list into our richest evidence source.
    """
    if not query:
        return []
    vids = await apify_youtube.search(query, max_results=12)
    if not vids:
        return []

    # Rank by recency desc (falls back to position). Grab transcripts for top 6.
    def _date_key(v: dict[str, Any]) -> str:
        return str(v.get("date") or "")
    ranked = sorted(vids, key=_date_key, reverse=True)
    transcript_targets = ranked[:6]

    async def _with_transcript(v: dict[str, Any]) -> dict[str, Any]:
        vid = v.get("video_id")
        transcript = ""
        if vid:
            # fetch_async tries the free API first, then falls back to
            # Apify — Railway IPs are typically blocked for the free API.
            transcript = await yt_transcript.fetch_async(vid, max_chars=4000)
        # Fallback chain: transcript → description → title.
        # Titles alone are useful — they often contain AED amounts,
        # location, partner names — so pass them as excerpt when we
        # have nothing better. The LLM downstream can still mine them.
        title = (v.get("title") or "").strip()
        excerpt = transcript or (v.get("description") or "").strip() or f"[YouTube title only] {title}"
        return {
            "source": f"youtube:{v.get('channel') or 'unknown'}",
            "source_url": v.get("url"),
            "title": title,
            "excerpt": excerpt[:2000],
            "raw_json": {
                "channel": v.get("channel"),
                "views": v.get("views"),
                "date": v.get("date"),
                "video_id": vid,
                "has_transcript": bool(transcript),
            },
        }

    enriched = await asyncio.gather(
        *[_with_transcript(v) for v in transcript_targets],
        return_exceptions=False,
    )
    # Remaining vids (beyond top 6) included with title as excerpt fallback.
    rest = [
        {
            "source": f"youtube:{v.get('channel') or 'unknown'}",
            "source_url": v.get("url"),
            "title": (v.get("title") or "").strip(),
            "excerpt": (
                (v.get("description") or "").strip()
                or f"[YouTube title only] {(v.get('title') or '').strip()}"
            )[:500],
            "raw_json": {
                "channel": v.get("channel"),
                "views": v.get("views"),
                "date": v.get("date"),
                "video_id": v.get("video_id"),
                "has_transcript": False,
            },
        }
        for v in ranked[6:]
    ]
    return enriched + rest


def _existing_signals_block(entities: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull any already-stored signals matching the developer or project."""
    dev = (entities.get("developer_slug") or "").lower().strip() or None
    project = (entities.get("project") or "").lower().strip() or None
    if not dev and not project:
        return []
    since = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    q = (
        sb.client()
        .table("signals")
        .select("id,cluster_key,headline,dek,urls,priority,confidence,entities,last_seen_at,dev_slug")
        .gte("last_seen_at", since)
        .eq("archived", False)
    )
    if dev:
        q = q.eq("dev_slug", dev)
    rows = q.limit(60).execute().data or []
    # If we have a project name, filter rows that mention it in headline/dek/entities.
    if project:
        needle = project.lower()
        rows = [
            r
            for r in rows
            if needle in (r.get("headline") or "").lower()
            or needle in (r.get("dek") or "").lower()
            or any(needle in (e or "").lower() for e in (r.get("entities") or []))
        ]
    out: list[dict[str, Any]] = []
    for r in rows:
        urls = r.get("urls") or []
        out.append(
            {
                "source": f"signals:id={r.get('id')}",
                "source_url": urls[0] if urls else None,
                "title": r.get("headline"),
                "excerpt": r.get("dek"),
                "raw_json": {
                    "cluster_key": r.get("cluster_key"),
                    "priority": r.get("priority"),
                    "confidence": r.get("confidence"),
                    "last_seen_at": r.get("last_seen_at"),
                    "dev_slug": r.get("dev_slug"),
                    "all_urls": urls,
                },
            }
        )
    return out


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_urls: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        url = (it.get("source_url") or "").strip().lower()
        key = url or f"{it.get('source')}|{(it.get('title') or '').lower()[:120]}"
        if key in seen_urls:
            continue
        seen_urls.add(key)
        out.append(it)
    return out


async def gather(entities: dict[str, Any], *, cap: int = 40) -> list[dict[str, Any]]:
    """Fan out across Tavily + Apify-YT + existing signals. Returns ≤cap items."""
    query = _q(entities)
    log.info("[spider] start query=%r", query[:100])
    tavily, yt = await asyncio.gather(
        _tavily_block(query), _yt_block(query), return_exceptions=True
    )
    items: list[dict[str, Any]] = []
    for block_name, block in (("tavily", tavily), ("yt", yt)):
        if isinstance(block, Exception):
            log.warning("[spider:%s] raised: %s", block_name, block)
            continue
        items.extend(block)
    # Deterministic block (no network) — always runs.
    items.extend(_existing_signals_block(entities))
    deduped = _dedupe(items)
    log.info(
        "[spider] gathered tavily=%d yt=%d signals=%d deduped=%d",
        len(tavily) if not isinstance(tavily, Exception) else 0,
        len(yt) if not isinstance(yt, Exception) else 0,
        len(deduped) - (len(tavily) if not isinstance(tavily, Exception) else 0)
        - (len(yt) if not isinstance(yt, Exception) else 0),
        len(deduped),
    )
    return deduped[:cap]
