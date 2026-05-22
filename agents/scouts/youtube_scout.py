"""YouTube scout — curated Dubai RE broker channels.

v2 replaces the Google-News-with-site:youtube hack. We now:
  1. Scan a curated list of broker + developer channels via free RSS
  2. Pull the full transcript for each new video via youtube-transcript-api
  3. Pre-filter transcripts on developer/RE vocabulary so only videos
     that actually discuss a launch / price / broker move get scored

No YT Data API key needed. youtube-transcript-api is a free community
lib; Apify is only used for on-demand search during the investigation
layer, not for the daily sweep.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx

log = logging.getLogger(__name__)


# Curated list — either direct channel_id (UC...) or handle URL.
# Handles get resolved to channel_ids at first use and cached.
# Add/remove as the MD signals-of-interest evolve.
BROKER_SOURCES: list[str] = [
    # Brokers
    "https://www.youtube.com/@LewisAllsopp",
    "https://www.youtube.com/@MoEznDubai",
    "https://www.youtube.com/@MattGreenDubai",
    "https://www.youtube.com/@AhmadAlZaabi",
    "https://www.youtube.com/@danwilliamsdubai",
    "https://www.youtube.com/@NadimNasri",
    "https://www.youtube.com/@ZachAminDubai",
    "https://www.youtube.com/@DrivenProperties",
    "https://www.youtube.com/@famproperties",
    "https://www.youtube.com/@luxhabitat",
    "https://www.youtube.com/@DubaiSothebys",
    "https://www.youtube.com/@AllsoppAllsopp",
    # Developers (official channels often telegraph launches)
    "https://www.youtube.com/@EmaarDubaiOfficial",
    "https://www.youtube.com/@DamacProperties",
    "https://www.youtube.com/@AldarProperties",
    "https://www.youtube.com/@SobhaRealty",
]


# 2026-04-29: themed YouTube channels. Each (channel_url, theme) ships
# `category_hint=theme` in raw_json so signals route to the right themed
# page on the dashboard regardless of the classifier's bias.
THEMED_CHANNELS: list[tuple[str, str]] = [
    # ─── Tech & AI in RE — proptech / contech / construction-tech ───
    ("https://www.youtube.com/@procoretech",         "tech"),
    ("https://www.youtube.com/@autodesk",            "tech"),
    ("https://www.youtube.com/@constructiondive",    "tech"),
    ("https://www.youtube.com/@TheBetterBuilders",   "tech"),
    ("https://www.youtube.com/@ICONbuild",           "tech"),
    # ─── Construction Economics — material price commentary + supply chain ───
    ("https://www.youtube.com/@ENRConstruction",     "materials"),
    ("https://www.youtube.com/@FastmarketsRMI",      "materials"),
    # ─── Global RE Pulse — luxury / prime market tours ───
    ("https://www.youtube.com/@knightfrank",         "global_prime"),
    ("https://www.youtube.com/@savillstv",           "global_prime"),
    ("https://www.youtube.com/@christiesinc",        "global_prime"),
    # ─── Capital Markets / HNWI advisory ───
    ("https://www.youtube.com/@HenleyPartners",      "capital_flow"),
]


RELEVANCE_RX = re.compile(
    r"\b("
    r"emaar|damac|aldar|nakheel|meraas|binghatti|azizi|omniyat|ellington|"
    r"deyaar|modon|sobha|dubai holding|"
    r"off.?plan|pre.?launch|launch|eoi|"
    r"payment plan|post.?handover|handover|delivery|"
    r"dld|rera|escrow|oqood|"
    r"psf|price per square|per sqft|starting (at|from)|aed \d|\bmillion\b|\bbn\b|\bbillion\b|"
    r"commission|broker|"
    r"reem island|palm jumeirah|downtown dubai|business bay|dubai marina|"
    r"dubai hills|jvc|jvt|damac hills|creek harbour|maritime city|"
    r"mbr city|meydan|arjan|dubailand|al furjan|tilal al ghaf|"
    # 2026-04-29 widen for themed channels (proptech / materials / prime / hnwi):
    r"proptech|contech|fractional|tokeniz|tokenis|"
    r"ai design|digital twin|bim|modular|3d.?print|prefab|robot|"
    r"funding round|series [a-c]|seed round|"
    r"rebar|steel price|cement|copper|aluminum|"
    r"shipping rate|lead time|hvac|elevator|generator|"
    r"prime|hnwi|wealth|migration|residency|"
    r"yield|spread|sukuk|treasury|eibor"
    r")\b",
    re.IGNORECASE,
)


_VIDEO_ID_RX = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([a-zA-Z0-9_-]{11})")


def _extract_video_id(url: str) -> str | None:
    m = _VIDEO_ID_RX.search(url)
    return m.group(1) if m else None


async def _resolve_channel_id(client: httpx.AsyncClient, source: str) -> str | None:
    """Resolve a handle URL (or passthrough a channel_id) to UC...."""
    if source.startswith("UC"):
        return source
    if "/channel/" in source:
        m = re.search(r"/channel/(UC[\w-]+)", source)
        return m.group(1) if m else None
    try:
        r = await client.get(source, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)
        r.raise_for_status()
        m = re.search(r'"channelId":"(UC[\w-]+)"', r.text)
        return m.group(1) if m else None
    except Exception as e:
        log.warning("[youtube] resolve failed for %s: %s", source, e)
        return None


def _rss_url(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "youtube"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _fetch_transcript_sync(video_id: str) -> str:
    """Blocking transcript fetch — run in a thread from the async scout."""
    from tools.yt_transcript import fetch_sync
    return fetch_sync(video_id, max_chars=4000)


async def _process_channel(
    client: httpx.AsyncClient, source: str, theme: str | None = None,
) -> list[dict[str, Any]]:
    channel_id = await _resolve_channel_id(client, source)
    if not channel_id:
        return []
    try:
        resp = await client.get(_rss_url(channel_id), timeout=10.0)
        resp.raise_for_status()
    except Exception as e:
        log.warning("[youtube:%s] RSS fetch failed: %s", channel_id, e)
        return []
    parsed = feedparser.parse(resp.text)
    # channel RSS gives 15 most recent; process the 8 newest to keep cost down
    entries = parsed.entries[:8]
    channel_title = (parsed.feed.get("title") if parsed.feed else None) or channel_id

    items: list[dict[str, Any]] = []
    for e in entries:
        title = (getattr(e, "title", "") or "").strip()[:300]
        url = (getattr(e, "link", "") or "").strip()
        if not title or not url:
            continue
        vid = _extract_video_id(url) or (getattr(e, "yt_videoid", None))
        date = getattr(e, "published", None) or getattr(e, "updated", None)
        # Transcript (blocking → offload).
        transcript = ""
        if vid:
            transcript = await asyncio.to_thread(_fetch_transcript_sync, vid)
        body = f"{title} {transcript[:2000]}"
        if not RELEVANCE_RX.search(body):
            continue
        raw_json: dict[str, Any] = {
            "channel_id": channel_id,
            "channel_title": channel_title,
            "video_id": vid,
            "date": date,
            "has_transcript": bool(transcript),
            "transcript": transcript[:4000] or None,
        }
        if theme:
            # 2026-04-29: themed channels (proptech / materials / prime /
            # capital_flow) ship category_hint so ingest routes them to
            # the right page without classifier dependency.
            raw_json["category_hint"] = theme
        items.append(
            {
                "source": f"youtube:{channel_title}",
                "source_url": url,
                "title": title,
                "summary": transcript[:600] if transcript else (getattr(e, "summary", "") or "")[:600],
                "raw_json": raw_json,
                "dedup_key": _dedup_key(title, url),
            }
        )
    log.info(
        "[youtube:%s] entries=%d kept=%d", channel_title, len(entries), len(items)
    )
    return items


async def run(limit: int = 40) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Legacy broker/dev channels — no theme override (classifier decides).
        # Themed channels — ship category_hint per channel.
        broker_jobs = [_process_channel(client, s) for s in BROKER_SOURCES]
        themed_jobs = [_process_channel(client, s, theme=t) for s, t in THEMED_CHANNELS]
        batches = await asyncio.gather(
            *broker_jobs, *themed_jobs,
            return_exceptions=True,
        )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            log.warning("[youtube] channel batch raised: %s", b)
            continue
        flat.extend(b)

    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        dedup.append(it)

    log.info("[youtube] total=%d deduped=%d", len(flat), len(dedup))
    return dedup[:limit]
