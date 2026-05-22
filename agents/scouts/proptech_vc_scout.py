"""PropTech VC + YC + GitHub trending scout (Round-2, 2026-04-29).

Tier-1 PropTech / ConTech VCs telegraph their theses through portfolio
moves — Fifth Wall, MetaProp, a16z, NEA, Brookfield. YC's Demo Days
and individual launches surface seed-stage proptech 6-12 months before
mainstream coverage. GitHub trending in proptech / contech topics shows
which open-source primitives are getting traction.

These three together are the EARLIEST tech-ecosystem layer — even
earlier than TechCrunch / Crunchbase (which the existing proptech_scout
already covers).

Three sub-channels in one scout:
  1. VC portfolio moves     — Fifth Wall, MetaProp, a16z, Brookfield
  2. YC launches + Demo Day — Hacker News + ycombinator.com + the YC
                              launches subdomain
  3. GitHub trending        — proptech / contech / construction-tech
                              topics

Stamp:
  - source: "proptech_vc:<channel>"
  - category_hint: "tech"   (routes to Tech & AI page)
  - tier_hint:     "press" (VC announcements / GH trending) or
                   "social" (YC launches / HN posts)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from tools import tavily_search

log = logging.getLogger(__name__)


# ─── 1. VC portfolio moves (Tavily) ───
VC_QUERY = (
    'PropTech ConTech "Fifth Wall" OR "MetaProp" OR "a16z" OR "NEA" OR '
    '"Brookfield" portfolio invest funding 2026 real estate construction'
)
VC_DOMAINS = [
    "fifthwall.com", "metaprop.com", "a16z.com", "nea.com",
    "brookfield.com", "techcrunch.com", "crunchbase.com",
    "news.crunchbase.com", "venturebeat.com",
]


# ─── 2. YC launches + Demo Day (Tavily) ───
YC_QUERY = (
    "Y Combinator YC PropTech ConTech construction real estate launch demo day 2026"
)
YC_DOMAINS = [
    "ycombinator.com", "news.ycombinator.com",
    "techcrunch.com", "crunchbase.com",
]


# ─── 3. GitHub trending (direct) ───
# GitHub publishes daily trending repos; we can fetch the JSON-ish HTML
# and filter to topics that map to proptech/contech.
GITHUB_TRENDING_URL = "https://github.com/trending?since=weekly"
GITHUB_TOPIC_TERMS = (
    "proptech", "contech", "construction", "real-estate",
    "bim", "digital-twin", "modular", "prefab", "iot-residential",
    "smart-home", "tokenized", "fractional-ownership",
)


MAX_AGE_DAYS = 30


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "proptech_vc"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(ts: Any) -> bool:
    if not ts:
        return True
    try:
        s = str(ts).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


async def _fetch_tavily(channel_key: str, label: str, query: str,
                       domains: list[str], tier: str) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            query, max_results=5, topic="news",
            search_depth="basic",
            include_domains=domains,
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[proptech_vc:%s] tavily failed: %s", channel_key, e)
        return []
    items: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        items.append({
            "source":     f"proptech_vc:{channel_key}",
            "source_url": url,
            "title":      f"{label} · {title[:240]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "channel":       channel_key,
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "tech",
                "tier_hint":     tier,
                "region":         "other",
                "country_code":   None,
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def _fetch_github_trending() -> list[dict[str, Any]]:
    """Pull GitHub's weekly trending list, filter to proptech-adjacent topics."""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                GITHUB_TRENDING_URL,
                headers={"User-Agent": "Sobha MDI proptech-vc scout"},
                timeout=12.0,
            )
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        log.warning("[proptech_vc:github] fetch failed: %s", e)
        return []

    # Light HTML scrape — repo blocks have <h2 class="h3 lh-condensed">
    # containing the repo path; descriptions are in <p class="col-9 ...">.
    import re as _re
    repo_rx = _re.compile(
        r'<h2[^>]*class="h3 lh-condensed"[^>]*>\s*<a[^>]+href="(/[^"]+)"',
        _re.IGNORECASE,
    )
    desc_rx = _re.compile(
        r'<p[^>]*class="col-9 [^"]*"[^>]*>(.*?)</p>',
        _re.IGNORECASE | _re.DOTALL,
    )
    matches = list(repo_rx.finditer(html))
    descs   = [d.group(1).strip() for d in desc_rx.finditer(html)]
    out: list[dict[str, Any]] = []
    for i, m in enumerate(matches[:25]):
        path = m.group(1).strip()
        repo_url = f"https://github.com{path}"
        desc = descs[i].replace("\n", " ").strip() if i < len(descs) else ""
        haystack = (path + " " + desc).lower()
        # Filter to proptech-adjacent topics.
        if not any(t in haystack for t in GITHUB_TOPIC_TERMS):
            continue
        title = f"GitHub trending · {path.lstrip('/')}"
        out.append({
            "source":     "proptech_vc:github_trending",
            "source_url": repo_url,
            "title":      title[:300],
            "summary":    (desc or "Trending PropTech / ConTech repository this week.")[:600],
            "raw_json": {
                "channel":       "github_trending",
                "repo_path":     path,
                "category_hint": "tech",
                "tier_hint":     "press",
                "region":         "other",
                "country_code":   None,
            },
            "dedup_key": _dedup_key(title, repo_url),
        })
    return out


async def run(limit: int = 18) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    vc, yc, gh = await asyncio.gather(
        _fetch_tavily("vc", "PropTech VC move", VC_QUERY, VC_DOMAINS, tier="press"),
        _fetch_tavily("yc", "YC launch", YC_QUERY, YC_DOMAINS, tier="social"),
        _fetch_github_trending(),
        return_exceptions=True,
    )
    for batch in (vc, yc, gh):
        if isinstance(batch, Exception):
            log.warning("[proptech_vc] batch raised: %s", batch)
            continue
        out.extend(batch or [])
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in out:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        deduped.append(it)
    log.info("[proptech_vc] total=%d deduped=%d", len(out), len(deduped))
    return deduped[:limit]
