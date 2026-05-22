"""AI-PropTech / ConTech scout — the headline 2026-04-28 stakeholder ask.

"Tech and AI News related to real estate to be added — both PropTech
and ConTech (construction tech)."

The existing gnews + gdelt scouts have a single broad PropTech query
each which gets buried under DLD-launch noise. This scout is dedicated:
  - PropTech funding rounds, M&A, IPOs (MENA + global)
  - ConTech innovation: modular, robotics, 3D-print, prefab residential
  - AI in design / BIM / digital-twin / generative-architecture
  - Tokenized real estate moves (Dubai pilots are the bellwether)
  - Smart-home + IoT standards

Sources: TechCrunch (domain-targeted), Crunchbase News, The Verge,
Wired, MENA-bytes, Wamda (MENA startups), MIT Tech Review,
Construction Dive (US ConTech wire), Dezeen + ArchDaily for AI-design
crossover, Building Design + Construction.

Free under existing Tavily plan. ~6 queries per run.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from tools import tavily_search

log = logging.getLogger(__name__)


PROPTECH_QUERIES: list[dict[str, Any]] = [
    {
        "label":   "PropTech funding global",
        "query":   "PropTech startup funding round series A B C 2026 real estate technology",
        "domains": ["techcrunch.com", "crunchbase.com", "news.crunchbase.com",
                    "theverge.com", "wired.com", "venturebeat.com",
                    "businessinsider.com", "forbes.com"],
        "subcat":  "proptech_funding",
    },
    {
        "label":   "PropTech MENA",
        "query":   "PropTech UAE Dubai Saudi MENA startup real estate technology funding 2026",
        "domains": ["wamda.com", "menabytes.com", "techcrunch.com",
                    "crunchbase.com", "news.crunchbase.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "gulfnews.com", "arabianbusiness.com"],
        "subcat":  "proptech_mena",
    },
    {
        "label":   "ConTech modular robotics 3D print",
        "query":   "construction technology modular robotics 3D printing prefab residential 2026",
        "domains": ["constructiondive.com", "enr.com", "bdcnetwork.com",
                    "techcrunch.com", "wired.com", "theverge.com",
                    "dezeen.com", "archdaily.com", "designboom.com"],
        "subcat":  "contech_innovation",
    },
    {
        "label":   "AI in architecture / BIM / digital twin",
        "query":   "AI generative architecture BIM digital twin design tools real estate 2026",
        "domains": ["dezeen.com", "archdaily.com", "designboom.com",
                    "wired.com", "theverge.com", "techcrunch.com",
                    "bdcnetwork.com", "constructiondive.com",
                    "technologyreview.com"],
        "subcat":  "ai_design",
    },
    {
        "label":   "Tokenized real estate Dubai pilot",
        "query":   "tokenized real estate blockchain Dubai pilot fractional ownership 2026",
        "domains": ["coindesk.com", "theblock.co", "decrypt.co",
                    "techcrunch.com", "thenationalnews.com",
                    "khaleejtimes.com", "gulfnews.com", "arabianbusiness.com"],
        "subcat":  "tokenized_re",
    },
    {
        "label":   "Smart-home IoT standards",
        "query":   "smart home IoT residential standard Matter Thread 2026 real estate",
        "domains": ["theverge.com", "wired.com", "cnet.com",
                    "techcrunch.com", "engadget.com",
                    "bdcnetwork.com", "constructiondive.com"],
        "subcat":  "smart_home",
    },
]


MAX_AGE_DAYS = 21


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "proptech"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True
    try:
        s = str(published_str).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


async def _fetch(query_def: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            query_def["query"],
            max_results=5,
            topic="news",
            search_depth="basic",
            include_domains=query_def["domains"],
            days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[proptech:%s] Tavily failed: %s", query_def["label"], e)
        return []

    out: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        # Region inference: MENA queries default to dubai bucket;
        # everything else is global ("other") — this content lives
        # on the Tech & AI page, not a region tab.
        region = "dubai" if query_def["subcat"] in ("proptech_mena", "tokenized_re") else "other"
        out.append(
            {
                "source": f"proptech:{query_def['subcat']}",
                "source_url": url,
                "title": f"Tech & AI · {title[:240]}",
                "summary": (r.get("content") or "")[:600],
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": r.get("published_date"),
                    "region": region,
                    "country_code": "AE" if region == "dubai" else None,
                    "category_hint": "tech",          # routes to tech_ai section
                    "subcategory": query_def["subcat"],
                    "topic": query_def["label"],
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
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


async def run(limit: int = 25) -> list[dict[str, Any]]:
    batches = await asyncio.gather(
        *[_fetch(q) for q in PROPTECH_QUERIES],
        return_exceptions=False,
    )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[proptech] queries=%d fetched=%d deduped=%d",
        len(PROPTECH_QUERIES), len(flat), len(deduped),
    )
    return deduped[:limit]
