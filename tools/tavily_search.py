"""Tavily web search — used by the investigation crew + several scouts
to go beyond our cached signal store. Tavily is built for agents:
returns clean, relevance-scored text snippets with URLs in one call.

Hardened 2026-04-26 against rate-limit + outage:
  1. In-process TTL cache (4h) — cuts repeat queries to ~half on a
     typical ingest run (industry_research_scout reuses the same firm-
     domain pairs across regions; trakheesi reuses intent prompts).
  2. Concurrency semaphore (8) — prevents bursts when ir_pages_scout
     fans out 45 queries × industry_research 31 pairs simultaneously.
  3. 429 retry with backoff — reads Retry-After if present, defaults
     to 10s, max 2 retries. Without this, a single 429 silently
     produced empty results that look identical to "no signal".
  4. Optional Brave Search fallback — env-gated by BRAVE_API_KEY.
     When Tavily 429s twice in a row, route the next call through
     Brave with a schema-mapped result shape so downstream scouts
     don't change.

Total Tavily calls per ingest run: ~131 across scouts (industry_research
~31 + linkedin ~50 + ir_pages ~45 + trakheesi ~5). At 2 runs/day = 262
calls/day. With cache: ~80/run = 160/day.

No-op when TAVILY_API_KEY is not set, so other flows never block.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx

from settings import settings

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# In-process cache + concurrency cap. Both are module-level so all
# callers share the budget; inside an async function the semaphore is
# created lazily to bind to the running event loop.
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL_S = 4 * 3600  # 4h — Tavily content is news; this matches our daily cadence
_SEMAPHORE: asyncio.Semaphore | None = None
_BRAVE_FAILS = 0  # counts consecutive Tavily 429s; flips fallback when ≥2
# 2026-04-30 audit-round: serialize updates to _BRAVE_FAILS so two
# concurrent 429 responses don't race and double-increment / both reset.
_BRAVE_LOCK = asyncio.Lock()


def _cache_key(query: str, body: dict[str, Any]) -> str:
    """Stable hash of (query, filter shape) — order-independent."""
    norm = {
        "q": query.strip().lower(),
        "depth": body.get("search_depth"),
        "topic": body.get("topic"),
        "max": body.get("max_results"),
        "include": tuple(sorted(body.get("include_domains") or [])),
        "exclude": tuple(sorted(body.get("exclude_domains") or [])),
        "days": body.get("days"),
    }
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()


def _get_cached(key: str) -> list[dict[str, Any]] | None:
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, val = hit
    if (time.time() - ts) > _CACHE_TTL_S:
        _CACHE.pop(key, None)
        return None
    return val


def _put_cache(key: str, value: list[dict[str, Any]]) -> None:
    _CACHE[key] = (time.time(), value)


async def _ensure_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(8)
    return _SEMAPHORE


async def _brave_fallback(
    query: str,
    *,
    max_results: int,
    include_domains: list[str] | None,
) -> list[dict[str, Any]]:
    """Drop-in fallback when Tavily is rate-limited. Maps Brave's
    response to Tavily's result shape so downstream scouts don't need
    to know which provider answered.
    """
    key = os.environ.get("BRAVE_API_KEY", "").strip()
    if not key:
        return []
    params: dict[str, Any] = {"q": query, "count": max_results}
    if include_domains:
        # Brave uses `site:` operator
        site_filter = " OR ".join(f"site:{d}" for d in include_domains)
        params["q"] = f"({query}) ({site_filter})"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                BRAVE_URL,
                params=params,
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("brave fallback failed: %s", e)
        return []
    web_results = (data.get("web") or {}).get("results") or []
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": r.get("description") or "",
            "score": 0.5,  # Brave doesn't expose a relevance score
            "published_date": r.get("page_age"),
        }
        for r in web_results[:max_results]
    ]


async def search(
    query: str,
    *,
    max_results: int = 10,
    search_depth: str = "advanced",  # 'basic' | 'advanced' — advanced reads deeper per page
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    topic: str = "general",           # 'general' | 'news'
    days: int | None = None,           # for topic='news', max age in days
) -> list[dict[str, Any]]:
    """Return [{title, url, content, score, published_date?}, ...]. Empty list on failure.

    Cached + concurrency-capped + 429-aware. Falls back to Brave when
    BRAVE_API_KEY is set and Tavily 429s twice in a row.
    """
    global _BRAVE_FAILS
    key = settings.tavily_api_key
    if not key:
        log.info("tavily: key not set — search skipped for %r", query[:80])
        return []

    body: dict[str, Any] = {
        "api_key": key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "topic": topic,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    if include_domains:
        body["include_domains"] = include_domains
    if exclude_domains:
        body["exclude_domains"] = exclude_domains
    if topic == "news" and days:
        body["days"] = days

    ck = _cache_key(query, body)
    cached = _get_cached(ck)
    if cached is not None:
        return cached

    sem = await _ensure_semaphore()
    async with sem:
        # Brave fallback path: when Tavily has 429'd repeatedly this run,
        # skip Tavily entirely until the next process restart.
        if _BRAVE_FAILS >= 2 and os.environ.get("BRAVE_API_KEY"):
            results = await _brave_fallback(
                query, max_results=max_results, include_domains=include_domains,
            )
            _put_cache(ck, results)
            return results

        # Tavily path with one 429 retry.
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(TAVILY_URL, json=body)
                    if resp.status_code == 429:
                        wait_s = 10.0
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                wait_s = min(float(ra), 30.0)
                            except ValueError:
                                pass
                        log.info(
                            "tavily 429 — backoff %.1fs then retry (attempt %d/2)",
                            wait_s, attempt + 1,
                        )
                        if attempt == 0:
                            await asyncio.sleep(wait_s)
                            continue
                        # Two 429s in a row — flip Brave fallback for the run.
                        async with _BRAVE_LOCK:
                            _BRAVE_FAILS += 1
                        results = await _brave_fallback(
                            query, max_results=max_results,
                            include_domains=include_domains,
                        )
                        _put_cache(ck, results)
                        return results
                    resp.raise_for_status()
                    data = resp.json()
                # Reset 429 counter on success
                async with _BRAVE_LOCK:
                    _BRAVE_FAILS = 0
                results = data.get("results") or []
                _put_cache(ck, results)
                return results
            except Exception as e:
                log.warning("tavily search failed (%s): %s", query[:60], e)
                if attempt == 0:
                    await asyncio.sleep(2.0)
                    continue
                # Final failure — try Brave fallback if available
                results = await _brave_fallback(
                    query, max_results=max_results,
                    include_domains=include_domains,
                )
                _put_cache(ck, results)
                return results

    return []
