"""On-demand YouTube search via Apify.

Used by the investigation spider to find every YT video mentioning a
project/developer/partner — not just uploads from our curated channel
list. Complements agents/scouts/youtube_scout.py which scans known
channels daily.

Returns [] when APIFY_TOKEN is unset or on any failure.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from tools.apify_client import run_actor

log = logging.getLogger(__name__)

ACTOR = "streamers~youtube-scraper"


async def search(query: str, *, max_results: int = 15) -> list[dict[str, Any]]:
    """Search YouTube for videos matching `query` and return compact records.

    Schema: [{title, url, channel, views, date, description, video_id}]
    """
    if not query or not query.strip():
        return []
    raw = await run_actor(
        ACTOR,
        {
            "searchQueries": [query],
            "maxResults": max_results,
            "maxResultsShorts": 0,
            "maxResultStreams": 0,
        },
        timeout_seconds=180,
    )
    items: list[dict[str, Any]] = []
    for v in raw:
        url = (v.get("url") or "").strip()
        title = (v.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        items.append(
            {
                "title": title,
                "url": url,
                "channel": (v.get("channelName") or v.get("channel") or "").strip(),
                "views": v.get("viewCount"),
                "date": v.get("date") or v.get("uploadDate"),
                "description": (v.get("description") or "")[:500],
                "video_id": v.get("id") or v.get("videoId"),
            }
        )
    log.info("[apify-yt] query=%r results=%d", query[:60], len(items))
    return items


def hash_video(url: str) -> str:
    return hashlib.sha256((url or "").encode()).hexdigest()[:16]
