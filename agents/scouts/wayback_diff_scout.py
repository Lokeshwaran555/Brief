"""Wayback Machine sitemap-diff scout — competitor URL discovery.

When a developer adds a new project page to their corporate site, the
URL appears in Wayback's CDX index within hours. Diffing the captured
URLs against our intel_raw history surfaces every new page —
including pre-launch microsite paths that don't get press coverage.

Strategy: query CDX `/cdx/search/cdx` for each tracked developer's
domain, scoped to the last 14 days. CDX returns one row per snapshot
(urlkey, timestamp, original URL, statuscode). We dedupe on
(domain, original URL) — first snapshot of a URL becomes a signal;
re-captures of an already-seen URL are dropped at the dedup_key
layer in Supabase.

Scout-side staleness filter: only emit URLs whose first capture is
within the last 14 days. Older URLs are existing-page noise.

Free, no key, no rate limit beyond polite use.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)


CDX_URL = (
    "https://web.archive.org/cdx/search/cdx?"
    "url={domain}/*&from={frm}&to={to}&"
    "output=json&filter=statuscode:200&filter=mimetype:text/html&"
    "collapse=urlkey&limit=200"
)


# Developer domain → (region, country_code, dev_slug). Domain values
# WITHOUT scheme — CDX expects bare host.
DOMAINS: list[dict[str, Any]] = [
    {"domain": "emaar.com",            "region": "dubai",     "country_code": "AE", "dev_slug": "emaar"},
    {"domain": "damacproperties.com",  "region": "dubai",     "country_code": "AE", "dev_slug": "damac"},
    {"domain": "nakheel.com",          "region": "dubai",     "country_code": "AE", "dev_slug": "nakheel"},
    {"domain": "sobharealty.com",      "region": "dubai",     "country_code": "AE", "dev_slug": "sobha"},
    {"domain": "binghatti.com",        "region": "dubai",     "country_code": "AE", "dev_slug": "binghatti"},
    {"domain": "azizidevelopments.com","region": "dubai",     "country_code": "AE", "dev_slug": "azizi"},
    {"domain": "ellingtonproperties.ae","region": "dubai",    "country_code": "AE", "dev_slug": "ellington"},
    {"domain": "omniyat.com",          "region": "dubai",     "country_code": "AE", "dev_slug": "omniyat"},
    {"domain": "deyaar.ae",            "region": "dubai",     "country_code": "AE", "dev_slug": "deyaar"},
    {"domain": "meraas.com",           "region": "dubai",     "country_code": "AE", "dev_slug": "dubaih"},
    {"domain": "aldar.com",            "region": "abu_dhabi", "country_code": "AE", "dev_slug": "aldar"},
    {"domain": "modon.ae",             "region": "abu_dhabi", "country_code": "AE", "dev_slug": "modon"},
    {"domain": "mubadala.com",         "region": "abu_dhabi", "country_code": "AE", "dev_slug": "mubadala"},
    {"domain": "mirvac.com",           "region": "australia", "country_code": "AU", "dev_slug": "mirvac"},
    {"domain": "stockland.com.au",     "region": "australia", "country_code": "AU", "dev_slug": "stockland"},
    {"domain": "lendlease.com",        "region": "australia", "country_code": "AU", "dev_slug": "lendlease"},
]


# Path tokens that signal a project / launch page (not a listings
# index, news article, or career link). Loose heuristic — we filter
# AFTER CDX to keep the API calls cheap.
PROJECT_PATH_HINTS: tuple[str, ...] = (
    "/project", "/properties/", "/property/", "/development",
    "/communities/", "/community/", "/residences/", "/residence",
    "/tower", "/launch", "/coming-soon", "/master-plan",
    "/phase-", "/villa", "/apartment", "/penthouse",
)

# Path tokens that should be REJECTED — these are noise pages.
PATH_REJECT: tuple[str, ...] = (
    "/news/", "/blog/", "/career", "/jobs", "/contact",
    "/privacy", "/terms", "/sitemap", "/feed", "/rss",
    "/login", "/signup", "/cookie", "/legal",
    ".pdf", ".jpg", ".png", ".css", ".js", ".xml",
)


DAYS_BACK = 14
LIMIT_PER_DOMAIN = 30


def _dedup_key(domain: str, url: str) -> str:
    head = f"wayback:{domain}:{url}"
    return hashlib.sha256(head.encode()).hexdigest()


def _path_looks_project(url: str) -> bool:
    """Heuristic: keep URLs whose path mentions a project-style token
    AND doesn't match any reject token. Conservative — we'd rather miss
    a few real pages than flood signals with /careers and /contact.
    """
    p = urlparse(url).path.lower()
    if not p or p == "/":
        return False
    if any(r in p for r in PATH_REJECT):
        return False
    return any(h in p for h in PROJECT_PATH_HINTS)


def _parse_cdx_timestamp(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


async def _fetch(
    client: httpx.AsyncClient,
    dev: dict[str, Any],
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    frm = (today - timedelta(days=DAYS_BACK)).strftime("%Y%m%d")
    to = today.strftime("%Y%m%d")
    url = CDX_URL.format(domain=dev["domain"], frm=frm, to=to)
    async with sem:
        try:
            resp = await client.get(
                url,
                headers={"User-Agent": "Sobha MDI Wayback-diff scout (contact: ops@sobha.com)"},
                timeout=45.0,
            )
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            log.warning("[wayback:%s] CDX fetch failed: %s",
                        dev["domain"], type(e).__name__)
            return []

    if not rows or len(rows) < 2:
        return []

    # First row is the column header; data starts at index 1.
    header = rows[0]
    try:
        ts_idx = header.index("timestamp")
        url_idx = header.index("original")
    except ValueError:
        log.warning("[wayback:%s] unexpected CDX header: %s", dev["domain"], header)
        return []

    out: list[dict[str, Any]] = []
    for row in rows[1:]:
        try:
            timestamp = row[ts_idx]
            original_url = row[url_idx]
        except (IndexError, TypeError):
            continue
        if not original_url:
            continue
        if not _path_looks_project(original_url):
            continue
        captured_at = _parse_cdx_timestamp(timestamp)
        captured_iso = captured_at.isoformat() if captured_at else None

        path = urlparse(original_url).path
        title = f"Wayback: new page on {dev['domain']} — {path[:200]}"
        summary = (
            f"First Wayback capture of {original_url} within the last "
            f"{DAYS_BACK} days. Captured at {captured_iso or '?'}."
        )
        out.append(
            {
                "source": f"wayback:{dev['dev_slug']}",
                "source_url": original_url,
                "title": title[:300],
                "summary": summary[:600],
                "raw_json": {
                    "domain": dev["domain"],
                    "captured_at": captured_iso,
                    "url_path": path,
                    "wayback_replay_url": (
                        f"https://web.archive.org/web/{timestamp}/{original_url}"
                    ),
                    "region": dev["region"],
                    "country_code": dev["country_code"],
                    "dev_slug_hint": dev["dev_slug"],
                    "category_hint": "leak",
                    "published_date": captured_iso,
                },
                "dedup_key": _dedup_key(dev["domain"], original_url),
            }
        )
        if len(out) >= LIMIT_PER_DOMAIN:
            break
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
    # CDX is slow (~5-8s per query) and rate-limits aggressive
    # parallelism. Cap to 4 concurrent requests across all domains.
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch(client, d, sem) for d in DOMAINS],
            return_exceptions=False,
        )
    flat = [r for batch in batches for r in batch]
    deduped = _dedupe(flat)
    log.info("[wayback] domains=%d fetched=%d deduped=%d",
             len(DOMAINS), len(flat), len(deduped))
    return deduped[:limit]
