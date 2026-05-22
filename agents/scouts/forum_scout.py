"""Forum / blog scout — Bayut blog, Property Finder blog, ExpatForum.

Two channels of public-but-under-circulated chatter:
  - Bayut & Property Finder editorial (lead-gen pieces often telegraph
    pricing strategy weeks before listings)
  - ExpatForum Dubai property threads (residents talking about
    handover delays, service charge rows, broker behaviour)

All RSS-based, no auth needed. Pre-filtered for relevance.
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
from urllib.parse import quote as _urlquote

log = logging.getLogger(__name__)


def _gn(domain: str, keywords: str, *, gl: str = "AE", hl: str = "en", when: str = "30d") -> str:
    """Google News site-restricted RSS query. 2026-05-18 rewrite:
    Bayut blog RSS 404'd, PF blog returns HTML, ExpatForum RSS
    returns HTML viewer page. Hit Google's index instead."""
    q = f'site:{domain} ({keywords}) when:{when}'
    ceid = f"{gl}:{hl}"
    return f"https://news.google.com/rss/search?q={_urlquote(q)}&hl={hl}&gl={gl}&ceid={ceid}"


_KW_BLOG  = ('"real estate" OR property OR "pre-launch" OR "off-plan" OR '
             'Emaar OR DAMAC OR Aldar OR Nakheel OR Binghatti OR Azizi OR '
             '"payment plan" OR "branded residence" OR "Golden Visa" OR '
             'sukuk OR mortgage OR Cityscape')
_KW_FORUM = ('"handover" OR "service charge" OR "snag" OR "DLD" OR '
             '"escrow" OR "off-plan" OR "rental" OR "broker" OR '
             '"property" OR "Emaar" OR "DAMAC" OR "Aldar"')

FEEDS: list[dict[str, str]] = [
    {"tag": "bayut-blog", "url": _gn("bayut.com", _KW_BLOG)},
    {"tag": "pf-blog",    "url": _gn("propertyfinder.ae", _KW_BLOG)},
    {"tag": "expatforum", "url": _gn("expatforum.com", _KW_FORUM + ' Dubai')},
]


# Keep the threshold generous on blogs (editorial is mostly relevant
# to RE) but tight on ExpatForum (high noise — visa, schools, taxis).
RELEVANCE_RX = re.compile(
    r"\b("
    r"emaar|damac|aldar|nakheel|meraas|binghatti|azizi|omniyat|ellington|"
    r"deyaar|modon|sobha|dubai holding|"
    r"off.?plan|pre.?launch|eoi|expression of interest|"
    r"handover|payment plan|post.?handover|escrow|oqood|"
    r"dld|rera|fatf|golden visa|"
    r"service charge|maintenance fee|snag|defect|"
    r"sukuk|mortgage|ltv|"
    r"psf|price per square|per sqft|"
    r"commission|broker"
    r")\b",
    re.IGNORECASE,
)


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "forum"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


async def _fetch_one(
    client: httpx.AsyncClient, feed: dict[str, str]
) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Mozilla/5.0 (SobhaMDI/1.0 forum-scout)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[forum:%s] fetch failed: %s", feed["tag"], e)
        return []

    parsed = feedparser.parse(resp.text)
    items: list[dict[str, Any]] = []
    is_blog = feed["tag"].endswith("-blog")
    for e in parsed.entries[:30]:
        title = (getattr(e, "title", "") or "").strip()[:300]
        url = (getattr(e, "link", "") or "").strip()
        if not title or not url:
            continue
        summary = (getattr(e, "summary", "") or "")[:600]
        # Blogs: light filter (drop only if NO RE keyword anywhere).
        # Forums: must clearly mention a developer / RE term.
        hay = f"{title} {summary[:500]}"
        if not RELEVANCE_RX.search(hay):
            if not is_blog:
                continue
            # Even on blogs, skip if zero RE vocab — they post lifestyle too.
            continue
        date = (
            getattr(e, "published", None)
            or getattr(e, "updated", None)
            or None
        )
        items.append(
            {
                "source": f"forum:{feed['tag']}",
                "source_url": url,
                "title": title,
                "summary": summary,
                "raw_json": {
                    "feed_tag": feed["tag"],
                    "date": date,
                    "author": getattr(e, "author", None),
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
    log.info(
        "[forum:%s] entries=%d kept=%d", feed["tag"], len(parsed.entries), len(items)
    )
    return items


async def run(limit: int = 40) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(*[_fetch_one(client, f) for f in FEEDS])
    flat = [row for batch in batches for row in batch]

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        out.append(it)

    log.info("[forum] total=%d deduped=%d", len(flat), len(out))
    return out[:limit]
