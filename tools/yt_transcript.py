"""Shared transcript fetcher — used by both the YT scout (daily sweep
on curated channels) and the investigation spider (on-demand search).

Two-tier strategy: try the free `youtube-transcript-api` first (fast,
no cost). If that fails (YouTube blocks many cloud IPs including
Railway's), fall back to Apify's transcript actor (residential IPs,
small per-call cost, works from any cloud).

Sync helpers; callers should run them in a thread via asyncio.to_thread.
Async helper `fetch_async` uses Apify directly.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

APIFY_TRANSCRIPT_ACTOR = "pintostudio~youtube-transcript-scraper"


def fetch_sync(video_id: str, *, max_chars: int = 6000) -> str:
    """Free-tier transcript fetch. Returns '' on any failure — callers
    should treat empty string as 'no transcript available' and may retry
    via fetch_async (Apify-backed) if they need robustness.
    """
    if not video_id:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        chunks = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"]
        )
        return " ".join(c.get("text", "") for c in chunks if c.get("text"))[:max_chars]
    except Exception as e:
        log.debug("transcript unavailable (api) for %s: %s", video_id, e)
        return ""


async def fetch_async(video_id: str, *, max_chars: int = 6000) -> str:
    """Robust transcript fetch: youtube-transcript-api → Apify fallback.

    Use this on Railway / cloud deployments where the free API is
    IP-blocked. Cost: ~$0.01 per transcript via Apify.
    """
    import asyncio
    if not video_id:
        return ""
    # 1. Try free API first.
    free = await asyncio.to_thread(fetch_sync, video_id, max_chars=max_chars)
    if free:
        return free
    # 2. Apify fallback.
    try:
        from tools.apify_client import run_actor
        items = await run_actor(
            APIFY_TRANSCRIPT_ACTOR,
            {"videoUrl": f"https://www.youtube.com/watch?v={video_id}"},
            timeout_seconds=90,
        )
        if not items:
            return ""
        # Actor returns [{text, start, duration}, ...] OR a single
        # {transcript: "..."} object depending on version. Handle both.
        first = items[0] if isinstance(items, list) else items
        if isinstance(first, dict) and "transcript" in first:
            return str(first["transcript"])[:max_chars]
        # Fallback shape — concatenate `text` fields across all items.
        joined = " ".join(
            str(i.get("text", "")) for i in items if isinstance(i, dict) and i.get("text")
        )
        return joined[:max_chars]
    except Exception as e:
        log.debug("transcript unavailable (apify) for %s: %s", video_id, e)
        return ""
