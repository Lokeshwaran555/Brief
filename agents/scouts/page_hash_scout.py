"""Page-hash scout — competitor marketing-page change detection.

Replaces a third-party page-monitoring service (ChangeTower etc.).
Each ingest run we fetch a curated list of competitor URLs, compute a
content hash from the cleaned HTML, compare to the snapshot stored in
Supabase, and emit a signal whenever the hash changes.

Captures the same signal class ChangeTower would (price card removed,
new project added to listings page, payment-plan banner swapped) but
runs on our own infra, free.

Limitations:
  - Doesn't render JavaScript. SPA-only competitor sites (where the
    initial HTML is empty + JS hydrates) will look identical every
    run; we'd need a headless browser for those.
  - Fragile to noise: cache-buster query strings, CSRF tokens,
    timestamp footers all mutate without real content change. We
    strip <script>/<style>/HTML tags + collapse whitespace as the
    cheapest defense; some false-positive change signals will leak.

Schema dependency: db/migrations/011_page_snapshots.sql
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from tools import supabase_tool as sb

log = logging.getLogger(__name__)


# Tracked URLs — same set we'd have given ChangeTower. 36 URLs.
# Each entry: url, dev_slug, region, country_code.
URLS: list[dict[str, Any]] = [
    # ─── Dubai (12) ───────────────────────────────────────────
    {"url": "https://www.emaar.com/en/properties",                    "dev_slug": "emaar",     "region": "dubai", "country_code": "AE"},
    {"url": "https://www.damacproperties.com/en/properties",          "dev_slug": "damac",     "region": "dubai", "country_code": "AE"},
    {"url": "https://www.nakheel.com/en/communities",                 "dev_slug": "nakheel",   "region": "dubai", "country_code": "AE"},
    {"url": "https://www.sobharealty.com/en/projects",                "dev_slug": "sobha",     "region": "dubai", "country_code": "AE"},
    {"url": "https://www.binghatti.com/projects",                     "dev_slug": "binghatti", "region": "dubai", "country_code": "AE"},
    {"url": "https://www.azizidevelopments.com/en/projects",          "dev_slug": "azizi",     "region": "dubai", "country_code": "AE"},
    {"url": "https://www.ellingtonproperties.ae/en/properties",       "dev_slug": "ellington", "region": "dubai", "country_code": "AE"},
    {"url": "https://www.omniyat.com/projects",                       "dev_slug": "omniyat",   "region": "dubai", "country_code": "AE"},
    {"url": "https://www.deyaar.ae/en/projects",                      "dev_slug": "deyaar",    "region": "dubai", "country_code": "AE"},
    {"url": "https://www.meraas.com/en/communities",                  "dev_slug": "dubaih",    "region": "dubai", "country_code": "AE"},
    {"url": "https://www.dubaiholding.com/en/our-businesses",         "dev_slug": "dubaih",    "region": "dubai", "country_code": "AE"},
    {"url": "https://dubaiproperties.ae/en/our-projects",             "dev_slug": "dubaih",    "region": "dubai", "country_code": "AE"},
    # ─── Abu Dhabi (4) ────────────────────────────────────────
    {"url": "https://www.aldar.com/en/explore-aldar/projects",        "dev_slug": "aldar",     "region": "abu_dhabi", "country_code": "AE"},
    {"url": "https://www.modon.ae/en/projects",                       "dev_slug": "modon",     "region": "abu_dhabi", "country_code": "AE"},
    {"url": "https://www.mubadala.com/en/what-we-do/real-estate",     "dev_slug": "mubadala",  "region": "abu_dhabi", "country_code": "AE"},
    {"url": "https://www.ihcgroup.com/our-businesses",                "dev_slug": "ihc",       "region": "abu_dhabi", "country_code": "AE"},
    # ─── Australia (4) ────────────────────────────────────────
    {"url": "https://www.mirvac.com/residential",                     "dev_slug": "mirvac",    "region": "australia", "country_code": "AU"},
    {"url": "https://www.stockland.com.au/residential",               "dev_slug": "stockland", "region": "australia", "country_code": "AU"},
    {"url": "https://www.lendlease.com/au/projects/",                 "dev_slug": "lendlease", "region": "australia", "country_code": "AU"},
    {"url": "https://www.charterhall.com.au/properties",              "dev_slug": "charterhall","region": "australia","country_code": "AU"},
    # ─── US REITs (5) ─────────────────────────────────────────
    {"url": "https://www.camdenliving.com/apartments",                "dev_slug": "camden",    "region": "usa", "country_code": "US"},
    {"url": "https://www.maac.com/communities",                       "dev_slug": "maa",       "region": "usa", "country_code": "US"},
    {"url": "https://www.avalonbay.com/apartments",                   "dev_slug": "avalonbay", "region": "usa", "country_code": "US"},
    {"url": "https://www.essexapartmenthomes.com/apartments",         "dev_slug": "ess",       "region": "usa", "country_code": "US"},
    {"url": "https://www.equityapartments.com/apartments",            "dev_slug": "eqr",       "region": "usa", "country_code": "US"},
    # ─── Branded-residence brand pipelines (6) ────────────────
    {"url": "https://www.aman.com/residences",                        "dev_slug": None, "region": None, "country_code": None},
    {"url": "https://www.bulgarihotels.com/residences",               "dev_slug": None, "region": None, "country_code": None},
    {"url": "https://www.mandarinoriental.com/residences",            "dev_slug": None, "region": None, "country_code": None},
    {"url": "https://www.fourseasons.com/private-residences/",        "dev_slug": None, "region": None, "country_code": None},
    {"url": "https://www.cipriani.com/residences",                    "dev_slug": None, "region": None, "country_code": None},
    {"url": "https://www.ritzcarlton.com/en/residences",              "dev_slug": None, "region": None, "country_code": None},
]


# Strip JS/CSS/markup before hashing. Cheap defense against most
# false-positive change detections (analytics pixels, inline timestamps).
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
# Drop common noise patterns: cache-buster query params + ISO timestamps.
NOISE_PATTERNS = [
    re.compile(r"\?v=[0-9a-f]+", re.IGNORECASE),
    re.compile(r"\?t=\d+", re.IGNORECASE),
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
    re.compile(r"csrf[-_]token['\"]?\s*[:=]\s*['\"][^'\"]+", re.IGNORECASE),
    re.compile(r"nonce-[A-Za-z0-9_-]+"),
]


def _clean_for_hash(html: str) -> str:
    text = SCRIPT_STYLE_RE.sub(" ", html or "")
    text = TAG_RE.sub(" ", text)
    for pat in NOISE_PATTERNS:
        text = pat.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _dedup_key(url: str, content_hash: str) -> str:
    return hashlib.sha256(f"page_hash:{url}:{content_hash}".encode()).hexdigest()


# ─── Supabase helpers — table page_snapshots (migration 011) ──
def _get_snapshot(url: str) -> dict[str, Any] | None:
    try:
        rows = (
            sb.client()
            .table("page_snapshots")
            .select("url,content_hash,first_seen_at,last_changed_at")
            .eq("url", url)
            .limit(1)
            .execute()
            .data
        )
    except Exception as e:
        log.warning("[page_hash] snapshot read failed for %s: %s", url, e)
        return None
    return rows[0] if rows else None


def _upsert_snapshot(url: str, content_hash: str, preview: str, changed: bool) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "url": url,
        "content_hash": content_hash,
        "raw_text_preview": preview[:1000],
        "last_seen_at": now_iso,
    }
    if changed:
        payload["last_changed_at"] = now_iso
    try:
        sb.client().table("page_snapshots").upsert(payload, on_conflict="url").execute()
    except Exception as e:
        log.warning("[page_hash] snapshot upsert failed for %s: %s", url, e)


# ─── Per-URL fetch + diff ─────────────────────────────────────
async def _check_url(client: httpx.AsyncClient, entry: dict[str, Any]) -> dict[str, Any] | None:
    url = entry["url"]
    try:
        resp = await client.get(
            url,
            headers={
                # Some competitor sites 403 on default httpx UA. Pretend
                # to be a polite indexer.
                "User-Agent": (
                    "Mozilla/5.0 (compatible; SobhaMDI-page-hash/1.0; "
                    "+https://sobha-mdi.app)"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[page_hash:%s] fetch failed: %s", url, type(e).__name__)
        return None

    cleaned = _clean_for_hash(resp.text)
    if not cleaned:
        return None
    h = _hash(cleaned)

    prior = _get_snapshot(url)
    if prior and prior.get("content_hash") == h:
        # Unchanged — bump last_seen_at, emit nothing.
        _upsert_snapshot(url, h, cleaned[:1000], changed=False)
        return None

    is_first_capture = prior is None
    _upsert_snapshot(url, h, cleaned[:1000], changed=True)

    if is_first_capture:
        # Don't emit a signal for the very first time we see a page —
        # that's just baseline establishment, not a change.
        log.info("[page_hash] baseline captured for %s", url)
        return None

    # Real change. Emit an intel_raw row.
    title = f"Page change detected: {url}"
    summary_bits = [
        f"Marketing page change at {url}.",
        f"Prior hash: {prior.get('content_hash','?')[:12]}",
        f"New hash: {h[:12]}.",
    ]
    summary = " ".join(summary_bits)

    return {
        "source": "page_hash:change",
        "source_url": url,
        "title": title[:300],
        "summary": summary[:600],
        "raw_json": {
            "url": url,
            "domain": urlparse(url).netloc,
            "old_hash": prior.get("content_hash"),
            "new_hash": h,
            "first_seen_at": prior.get("first_seen_at"),
            "region": entry.get("region"),
            "country_code": entry.get("country_code"),
            "dev_slug_hint": entry.get("dev_slug"),
            "category_hint": "marketing_page_diff",
            "published_date": datetime.now(timezone.utc).isoformat(),
        },
        "dedup_key": _dedup_key(url, h),
    }


# Keep concurrency moderate — some competitor sites IP-throttle.
_SCOUT_CONCURRENCY = 6


async def run(limit: int = 30) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(_SCOUT_CONCURRENCY)

    async def _bounded(client: httpx.AsyncClient, entry: dict[str, Any]):
        async with sem:
            return await _check_url(client, entry)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(
            *[_bounded(client, e) for e in URLS],
            return_exceptions=False,
        )

    out = [r for r in results if r]
    log.info(
        "[page_hash] urls=%d changes_emitted=%d (baseline/unchanged=%d)",
        len(URLS), len(out), len(URLS) - len(out),
    )
    return out[:limit]
