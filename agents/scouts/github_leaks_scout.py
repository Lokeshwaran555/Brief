"""GitHub Code Search scout — leak detection for project-name candidates.

Marketing agencies, sales-portal vendors, and microsite builders push
staging code with project names embedded — sometimes pre-launch.
"Sobha Reserve" appeared in a GitHub commit before the press release.
Same pattern catches competitor pre-launches.

Strategy: query GitHub Code Search via the authenticated API for a
small set of project-name candidates per developer. We can't search
unauth — GitHub Code Search requires a Personal Access Token. The
scout fails-soft when GITHUB_TOKEN is not configured (returns []),
so it's safe to ship before the token is set.

Quota: code search rate limit is 30 queries/minute authenticated.
We send ~12 queries per run (one per dev). Trivial.

Setup once:
  1. Create a fine-grained Personal Access Token at
     github.com/settings/tokens?type=beta with read-only access to
     "Public repositories" (no write scopes needed).
  2. Set env var GITHUB_TOKEN on Railway.
  3. Restart service. No re-deploy needed.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)


GITHUB_SEARCH_URL = "https://api.github.com/search/code?q={q}&per_page={per_page}"
PER_QUERY_LIMIT = 5
GITHUB_TIMEOUT = 12.0


# Each query is a project-name candidate or developer name. Code
# search matches on file content, file path, and repo name. We
# include developer names alone (catches pre-launch microsite repos)
# and "developer + property" (catches sales-portal staging).
QUERIES: list[dict[str, Any]] = [
    # Tracked Dubai developers
    {"tag": "emaar",     "q": '"emaar properties" OR "emaar.com"',           "region": "dubai",     "country_code": "AE"},
    {"tag": "damac",     "q": '"damac properties" OR "damacproperties.com"', "region": "dubai",     "country_code": "AE"},
    {"tag": "nakheel",   "q": '"nakheel" "real estate"',                     "region": "dubai",     "country_code": "AE"},
    {"tag": "sobha",     "q": '"sobha realty" OR "sobhareality.com"',        "region": "dubai",     "country_code": "AE"},
    {"tag": "binghatti", "q": '"binghatti developers"',                      "region": "dubai",     "country_code": "AE"},
    {"tag": "azizi",     "q": '"azizi developments"',                        "region": "dubai",     "country_code": "AE"},
    {"tag": "ellington", "q": '"ellington properties"',                      "region": "dubai",     "country_code": "AE"},
    {"tag": "omniyat",   "q": '"omniyat" "dubai"',                           "region": "dubai",     "country_code": "AE"},
    # Abu Dhabi
    {"tag": "aldar",     "q": '"aldar properties" OR "aldar.com"',           "region": "abu_dhabi", "country_code": "AE"},
    {"tag": "modon",     "q": '"modon properties"',                          "region": "abu_dhabi", "country_code": "AE"},
    # Branded-residence partners — global pre-launch hint
    {"tag": "branded-aman",      "q": '"aman residences" "dubai" OR "uae"',  "region": None,        "country_code": None},
    {"tag": "branded-bulgari",   "q": '"bulgari residences" "dubai" OR "uae"', "region": None,      "country_code": None},
]


def _dedup_key(repo: str, path: str) -> str:
    head = f"github:{repo}:{path}"
    return hashlib.sha256(head.encode()).hexdigest()


async def _fetch(client: httpx.AsyncClient, query: dict[str, Any], token: str) -> list[dict[str, Any]]:
    url = GITHUB_SEARCH_URL.format(q=quote(query["q"]), per_page=PER_QUERY_LIMIT)
    try:
        resp = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "Sobha MDI GitHub-leaks scout (contact: ops@sobha.com)",
            },
            timeout=GITHUB_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[github:%s] search failed: %s", query["tag"], e)
        return []

    items = data.get("items") or []
    out: list[dict[str, Any]] = []
    for it in items:
        repo = (it.get("repository") or {}).get("full_name") or ""
        path = it.get("path") or ""
        html_url = it.get("html_url") or ""
        if not repo or not path:
            continue
        # Drop hits inside very-popular meta repos (awesome-lists,
        # crawled-data, etc.) — they're noise, not pre-launch leaks.
        repo_l = repo.lower()
        if any(x in repo_l for x in ("awesome-", "crawl-data", "common-crawl", "wikipedia", "warehouse")):
            continue

        title = f"GitHub leak [{query['tag']}]: {repo}/{path}"
        summary_bits = [
            f"Match for query: {query['q']}",
            f"Repo: {repo}",
            f"File: {path}",
        ]
        score = it.get("score")
        if score:
            summary_bits.append(f"Match score: {score:.2f}")
        out.append(
            {
                "source": f"github_leaks:{query['tag']}",
                "source_url": html_url,
                "title": title[:300],
                "summary": " · ".join(summary_bits)[:600],
                "raw_json": {
                    "query_tag": query["tag"],
                    "query": query["q"],
                    "repo": repo,
                    "path": path,
                    "score": score,
                    "region": query.get("region"),
                    "country_code": query.get("country_code"),
                    "category_hint": "leak",
                },
                "dedup_key": _dedup_key(repo, path),
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


async def run(limit: int = 20) -> list[dict[str, Any]]:
    token = os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        log.info("[github_leaks] GITHUB_TOKEN not configured — skipping run")
        return []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch(client, q, token) for q in QUERIES],
            return_exceptions=False,
        )

    flat = [r for batch in batches for r in batch]
    deduped = _dedupe(flat)
    log.info("[github_leaks] queries=%d fetched=%d deduped=%d",
             len(QUERIES), len(flat), len(deduped))
    return deduped[:limit]
