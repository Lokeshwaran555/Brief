"""Curated newsletter / RSS scout (Round-4, 2026-04-30).

Newsletters are pre-filtered intelligence — the curator has already
done the editorial work, so we trust the source and ingest every
recent entry. Each newsletter is hand-tagged with the topic that
matches its editorial beat, which routes the resulting signals to the
correct themed page automatically (via category_hint at promotion).

Coverage:
  - AI / ML research          → category_hint = tech
  - PropTech / RE Tech        → category_hint = tech
  - ConTech / Construction    → category_hint = materials
  - Architecture / Design     → category_hint = architecture
  - RE Markets / Macro        → category_hint = capital_markets
  - Geopolitics commentary    → category_hint = geopolitics

Strategy: hit each feed's RSS endpoint, take the most recent N entries
within a 14-day window, stamp the topic, dedup by (title, host).
No keyword filter — the newsletter IS the filter.

Free, no keys.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx

log = logging.getLogger(__name__)


# Each entry: {name, url, topic, category_hint, region?, country?}
# - topic is a short label for the source field stamp
# - category_hint maps directly to the themed-page routing in the keystone
# - region/country left None when global; downstream classifier infers
FEEDS: list[dict[str, Any]] = [
    # --- AI / ML research newsletters ---
    {"name": "Import AI",         "url": "https://jack-clark.net/feed/",
     "topic": "ai_research",      "category_hint": "tech"},
    {"name": "The Batch",         "url": "https://www.deeplearning.ai/the-batch/feed/",
     "topic": "ai_research",      "category_hint": "tech"},
    {"name": "Last Week in AI",   "url": "https://lastweekin.ai/feed",
     "topic": "ai_research",      "category_hint": "tech"},
    {"name": "Latent Space",      "url": "https://www.latent.space/feed",
     "topic": "ai_research",      "category_hint": "tech"},
    {"name": "AI Snake Oil",      "url": "https://www.aisnakeoil.com/feed",
     "topic": "ai_research",      "category_hint": "tech"},
    {"name": "Ben's Bites",       "url": "https://bensbites.beehiiv.com/feed",
     "topic": "ai_industry",      "category_hint": "tech"},
    {"name": "TLDR AI",           "url": "https://tldr.tech/api/rss/ai",
     "topic": "ai_industry",      "category_hint": "tech"},

    # --- PropTech / RE Tech newsletters ---
    # 2026-05-08 stakeholder feedback: US trade press was dominating
    # the visible /api/intel slice (78 items in 200-cap, all tagged
    # region=None falling into 'other'/'null' buckets and crowding
    # out Dubai/AD coverage). Two fixes:
    #   1. Tag region="usa" so they go to the USA tab, not 'other'
    #   2. Halve per-feed cap from 12 → 5/4 so US volume is bounded
    # The MD has 50 scouts; US trade press should not dominate.
    {"name": "Propmodo",          "url": "https://www.propmodo.com/feed/",
     "topic": "proptech",         "category_hint": "tech",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "The Real Deal",     "url": "https://therealdeal.com/feed/",
     "topic": "re_press",         "category_hint": "competitor",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "Bisnow",            "url": "https://www.bisnow.com/rss/national",
     "topic": "re_press",         "category_hint": "competitor",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "GlobeSt",           "url": "https://www.globest.com/rss",
     "topic": "re_press",         "category_hint": "competitor",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "Inman",             "url": "https://www.inman.com/feed/",
     "topic": "re_press",         "category_hint": "competitor",
     "region": "usa", "country": "US", "max_items": 3},

    # --- ConTech / Construction industry newsletters ---
    # US construction press also tagged region=usa + capped (was a
    # secondary contributor to USA dominance via 'other' bucket).
    {"name": "Construction Dive", "url": "https://www.constructiondive.com/feeds/news/",
     "topic": "contech",          "category_hint": "materials",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "ENR",               "url": "https://www.enr.com/rss/all",
     "topic": "construction_press", "category_hint": "materials",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "BD+C",              "url": "https://www.bdcnetwork.com/rss.xml",
     "topic": "construction_press", "category_hint": "materials",
     "region": "usa", "country": "US", "max_items": 3},
    {"name": "Construction Index","url": "https://www.theconstructionindex.co.uk/news/feeds/rss",
     "topic": "construction_press", "category_hint": "materials"},
    {"name": "AEC Magazine",      "url": "https://aecmag.com/feed/",
     "topic": "contech",          "category_hint": "materials"},
    {"name": "Architects Newspaper", "url": "https://www.archpaper.com/feed/",
     "topic": "architecture",     "category_hint": "architecture"},

    # --- RE Markets / Macro commentary ---
    {"name": "Wolf Street",       "url": "https://wolfstreet.com/feed/",
     "topic": "macro_commentary", "category_hint": "capital_markets"},
    {"name": "Calculated Risk",   "url": "https://www.calculatedriskblog.com/feeds/posts/default",
     "topic": "macro_commentary", "category_hint": "capital_markets"},
    {"name": "Mansion Global",    "url": "https://www.mansionglobal.com/feed",
     "topic": "global_prime",     "category_hint": "global_prime"},

    # --- Geopolitics with capital-flow / RE relevance ---
    {"name": "Noahpinion",        "url": "https://www.noahpinion.blog/feed",
     "topic": "geo_commentary",   "category_hint": "geopolitics"},

    # --- Tech / VC general (catch PropTech & ConTech funding mentions) ---
    {"name": "Stratechery (free)", "url": "https://stratechery.com/feed/",
     "topic": "tech_strategy",    "category_hint": "tech"},
]


MAX_AGE_DAYS = 14
MAX_PER_FEED = 12


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "newsletter"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published: Any) -> bool:
    if not published:
        return True
    try:
        s = str(published).rstrip("Z")
        # feedparser sometimes returns RFC822, sometimes ISO; try ISO first.
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


async def _fetch(client: httpx.AsyncClient, feed: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI newsletter scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[newsletter:%s] fetch failed: %s", feed["name"], e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    # Per-feed cap defaults to MAX_PER_FEED (12) but each FEED entry
    # can override via "max_items" — used 2026-05-08 to throttle US
    # trade press to 4-5 items each so they don't dominate the slice.
    cap = int(feed.get("max_items") or MAX_PER_FEED)
    for e in parsed.entries[:cap]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        summary = (getattr(e, "summary", "") or "").strip()
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue

        # Strip any inline HTML from summary so the bullet renderer doesn't
        # have to. feedparser leaves <p> wrappers and the occasional <img>.
        import re
        summary_clean = re.sub(r"<[^>]+>", "", summary)
        summary_clean = re.sub(r"\s+", " ", summary_clean).strip()[:600]

        slug = feed["name"].lower().replace(" ", "_").replace("+", "plus").replace("'", "")
        out.append({
            "source":     f"newsletter:{slug}",
            "source_url": link,
            "title":      f"{feed['name']} · {title[:240]}",
            "summary":    summary_clean,
            "raw_json": {
                "newsletter":   feed["name"],
                "topic":        feed.get("topic"),
                "published_date": published,
                "category_hint": feed.get("category_hint"),
                "region":       feed.get("region"),
                "country_code": feed.get("country"),
                "tier_hint":    "newsletter",  # curated > raw press
            },
            "dedup_key": _dedup_key(title, link),
        })
    return out


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        k = it["dedup_key"]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


async def run(limit: int = 60) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch(client, feed) for feed in FEEDS],
            return_exceptions=True,
        )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            continue
        flat.extend(b)
    deduped = _dedupe(flat)
    log.info(
        "[newsletters] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
