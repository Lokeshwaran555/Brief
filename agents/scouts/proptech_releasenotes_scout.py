"""PropTech release-notes scout — RealPage/Yardi/AppFolio/MRI changelog.

These four are the backbone of US multifamily ops; their product release
notes tell you what's becoming standard 12-18 months ahead. Useful for
US/Texas multifamily exposure but also a leading indicator of which
PropTech features will arrive in Dubai 2-3 years later.

Tavily-mediated against vendor docs domains; the changelog endpoints
exist but require auth or shift formats. Tavily handles the indexing.

Stamp:
  - source: "proptech_releasenotes:<vendor>"
  - category_hint: "tech"
  - tier_hint: "official"
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


VENDORS: list[dict[str, Any]] = [
    {
        "label":   "RealPage",
        "query":   "RealPage product release notes 2026 update feature announcement",
        "domains": ["realpage.com", "techcrunch.com", "constructiondive.com",
                    "bdcnetwork.com", "businesswire.com", "prnewswire.com"],
        "key":     "realpage",
    },
    {
        "label":   "Yardi",
        "query":   "Yardi product release notes 2026 update feature announcement",
        "domains": ["yardi.com", "techcrunch.com", "constructiondive.com",
                    "bdcnetwork.com", "businesswire.com", "prnewswire.com"],
        "key":     "yardi",
    },
    {
        "label":   "AppFolio",
        "query":   "AppFolio product release notes 2026 update feature property management",
        "domains": ["appfolio.com", "techcrunch.com", "constructiondive.com",
                    "bdcnetwork.com", "businesswire.com", "prnewswire.com"],
        "key":     "appfolio",
    },
    {
        "label":   "MRI Software",
        "query":   "MRI Software product release notes 2026 update feature real estate",
        "domains": ["mrisoftware.com", "techcrunch.com",
                    "constructiondive.com", "bdcnetwork.com",
                    "businesswire.com", "prnewswire.com"],
        "key":     "mri",
    },
]


MAX_AGE_DAYS = 60


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    domain = urlparse(url).netloc.lower() if url else "proptech_rn"
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


async def _fetch_one(v: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            v["query"], max_results=4,
            topic="news", search_depth="basic",
            include_domains=v["domains"], days=MAX_AGE_DAYS,
        )
    except Exception as e:
        log.warning("[proptech_rn:%s] tavily failed: %s", v["key"], e)
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
            "source":     f"proptech_releasenotes:{v['key']}",
            "source_url": url,
            "title":      f"PropTech rn · {v['label']} · {title[:200]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "vendor":        v["key"],
                "label":         v["label"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "tech",
                "tier_hint":     "official",
                "region":        "other",
                "country_code":  "US",
            },
            "dedup_key": _dedup_key(title, url),
        })
    return items


async def run(limit: int = 12) -> list[dict[str, Any]]:
    batches = await asyncio.gather(*[_fetch_one(v) for v in VENDORS],
                                   return_exceptions=True)
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            continue
        flat.extend(b)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        out.append(it)
    log.info("[proptech_releasenotes] vendors=%d fetched=%d deduped=%d",
             len(VENDORS), len(flat), len(out))
    return out[:limit]
