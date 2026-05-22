"""LinkedIn scout — Tavily with site:linkedin.com filter.

Apify has LinkedIn actors too but they need per-actor sign-in and
rate-limit aggressively. Tavily with `include_domains=['linkedin.com']`
indexes company posts + profile activity that surface via public
crawling, which covers the 80% case:
  - developer company-page posts (launch teasers, event invites)
  - exec profile activity (leadership moves, pipeline announcements)
  - broker posts (deal closings, commission chatter)

We run one Tavily query per tier-1 developer × 2 intent buckets
(announcements + hiring/leadership), dedup across queries, and return
raw signals the classifier can score like any other source.

Silently returns [] when TAVILY_API_KEY is unset.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from tools import tavily_search

log = logging.getLogger(__name__)


# (developer_name, developer_slug, region, country_code).
# Region drives dashboard tab routing; slug used for dedup against
# the Dubai DEV_MATCHERS in ingest_flow.
DEVS: list[tuple[str, str, str, str]] = [
    # ─── Dubai ───────────────────────────────────────────────────
    ("Emaar", "emaar", "dubai", "AE"),
    ("DAMAC", "damac", "dubai", "AE"),
    ("Nakheel", "nakheel", "dubai", "AE"),
    ("Meraas", "dubaih", "dubai", "AE"),
    ("Dubai Holding", "dubaih", "dubai", "AE"),
    ("Binghatti", "binghatti", "dubai", "AE"),
    ("Azizi", "azizi", "dubai", "AE"),
    ("Ellington", "ellington", "dubai", "AE"),
    ("Omniyat", "omniyat", "dubai", "AE"),
    ("Deyaar", "deyaar", "dubai", "AE"),
    # ─── Abu Dhabi ───────────────────────────────────────────────
    ("Aldar", "aldar", "abu_dhabi", "AE"),
    ("Modon", "modon", "abu_dhabi", "AE"),
    ("Mubadala", "mubadala", "abu_dhabi", "AE"),
    ("IHC", "ihc", "abu_dhabi", "AE"),
    ("ADQ", "adq", "abu_dhabi", "AE"),
    # ─── USA (Texas focus + nationwide multifamily REITs) ────────
    ("Camden Property Trust", "camden", "usa", "US"),
    ("Mid-America Apartment Communities", "maa", "usa", "US"),
    ("Mill Creek Residential", "millcreek", "usa", "US"),
    ("Tricon Residential", "tricon", "usa", "US"),
    ("Greystar", "greystar", "usa", "US"),
    # ─── Australia (Brisbane priority + nationwide developers) ───
    ("Mirvac", "mirvac", "australia", "AU"),
    ("Stockland", "stockland", "australia", "AU"),
    ("Lendlease", "lendlease", "australia", "AU"),
    ("Goodman Group", "goodman", "australia", "AU"),
    ("Charter Hall", "charterhall", "australia", "AU"),
]


# Per-region intent qualifier — appended to the dev query so the
# search stays localized. Empty string for global / no qualifier.
REGION_QUALIFIER = {
    "dubai": "Dubai",
    "abu_dhabi": "Abu Dhabi",
    "usa": "United States",
    "australia": "Australia",
}


QUERY_TEMPLATES: list[tuple[str, str]] = [
    # (tag, query_template with {dev} {qual})
    (
        "company-announce",
        '"{dev}" {qual} (launch OR unveils OR "pre-launch" OR EOI OR handover OR partnership OR "now selling" OR acquisition)',
    ),
    (
        "leadership-hiring",
        '"{dev}" {qual} ("Head of Sales" OR "VP Sales" OR "Sales Director" OR "Chief Commercial" OR "Managing Director" OR appointment OR hired)',
    ),
]


# 2026-04-29: Themed LinkedIn targets — proptech / materials / global RE
# prime brokers / capital-markets banks. Each entry has an explicit theme
# so the resulting signals route to the right themed page on the dashboard
# (no classifier dependency). region="other" by default since most are
# global; banks default to dubai region (UAE-resident).
# Tuple shape: (company, theme, region, country_code).
THEMED_TARGETS: list[tuple[str, str, str, str]] = [
    # ─── Tech & AI in RE ───
    ("Stake",            "tech", "dubai", "AE"),
    ("Smartcrowd",       "tech", "dubai", "AE"),
    ("Square Yards",     "tech", "other", "IN"),
    ("Property Finder",  "tech", "dubai", "AE"),
    ("Bayut",            "tech", "dubai", "AE"),
    ("Procore",          "tech", "other", "US"),
    ("Autodesk",         "tech", "other", "US"),
    ("Buildots",         "tech", "other", "IL"),
    # ─── Construction Economics — UAE suppliers + globals ───
    ("Emirates Steel",   "materials", "abu_dhabi", "AE"),
    ("Conmix",           "materials", "dubai",     "AE"),
    ("Lafarge",          "materials", "other",     "FR"),
    ("Holcim",           "materials", "other",     "CH"),
    ("Pinnacle Industries", "materials", "dubai",  "AE"),
    # ─── Global RE Pulse — prime brokers + HNWI advisors ───
    ("Knight Frank",     "global_prime", "other", "GB"),
    ("Savills",          "global_prime", "other", "GB"),
    ("Christie's International Real Estate", "global_prime", "other", "US"),
    ("Sotheby's International Realty",       "global_prime", "other", "US"),
    ("Henley & Partners","capital_flow",  "other", "AE"),
    # ─── Capital Markets — UAE banks publishing rate moves ───
    ("First Abu Dhabi Bank", "capital_markets", "abu_dhabi", "AE"),
    ("Emirates NBD",         "capital_markets", "dubai",     "AE"),
    ("ADCB",                 "capital_markets", "abu_dhabi", "AE"),
    ("ADIB",                 "capital_markets", "abu_dhabi", "AE"),
]


# Themed-target query — same shape as developer queries but tuned for
# the broader signal set: announcements + thought leadership + funding.
THEMED_TEMPLATE = (
    '"{dev}" {qual} (launch OR announcement OR funding OR "Series A" OR "Series B" OR partnership OR acquisition OR appointed OR "joins as" OR research OR report OR forecast)'
)


LI_DOMAINS = ["linkedin.com"]

# Hard date floor for LinkedIn scout. Tavily's general topic doesn't
# accept the `days` param, so we filter post-hoc by published_date.
# Posts older than this are dropped before they enter intel_raw — the
# MD doesn't want "Nakheel sold out" headlines from a year ago re-
# surfacing as fresh intel.
MAX_AGE_DAYS = 30

# LinkedIn activity / share / ugcPost URN IDs are 19-digit snowflake-
# style numbers whose upper 41 bits encode the post creation time in
# Unix milliseconds. We extract those from the URL because Tavily
# rarely returns a usable published_date for LinkedIn (LinkedIn strips
# OG/schema.org dates from public HTML to discourage scraping).
#
# Examples we want to match:
#   /posts/aldar-properties_launch-activity-7234567890123456789-XXXX
#   /feed/update/urn:li:activity:7234567890123456789/
#   /feed/update/urn:li:share:7234567890123456789
#   /feed/update/urn:li:ugcPost:7234567890123456789
_LI_SNOWFLAKE_RE = re.compile(
    r"(?:activity[:\-]|share[:\-]|ugcPost[:\-])(\d{17,19})",
    re.IGNORECASE,
)
# Sanity bounds for the decoded timestamp — anything outside this
# range came from a non-snowflake ID we shouldn't trust.
_LI_FLOOR_MS = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "linkedin"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _decode_url_timestamp(url: str) -> datetime | None:
    """Recover the post creation date from a LinkedIn URL.

    Returns aware UTC datetime, or None if no decodable activity ID
    or the decoded timestamp is implausible (before 2018, or in the
    future). Does NOT make any network call — purely structural.
    """
    if not url:
        return None
    m = _LI_SNOWFLAKE_RE.search(url)
    if not m:
        return None
    try:
        activity_id = int(m.group(1))
    except (TypeError, ValueError):
        return None
    ts_ms = activity_id >> 22
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if ts_ms < _LI_FLOOR_MS or ts_ms > now_ms + 86_400_000:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _is_within_age(published_date: Any) -> bool:
    """True only when we have a parseable date AND it falls within the
    age window. Unknown dates are NOT treated as fresh for LinkedIn —
    Tavily strips dates from old LinkedIn posts, so 'unknown' was
    letting 6+ month posts through. Caller should give every URL a
    chance to recover a date via _decode_url_timestamp() first.
    """
    if not published_date:
        return False
    try:
        s = str(published_date).rstrip("Z")
        dt = datetime.fromisoformat(s)
    except Exception:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - dt
    return age <= timedelta(days=MAX_AGE_DAYS)


async def _run_query(
    tag: str, query: str, *, region: str | None = None, country_code: str | None = None,
    theme: str | None = None,
) -> list[dict[str, Any]]:
    results = await tavily_search.search(
        query,
        max_results=6,
        topic="general",
        search_depth="basic",
        include_domains=LI_DOMAINS,
    )
    items: list[dict[str, Any]] = []
    dropped_old = 0
    dropped_undated = 0
    recovered_from_url = 0
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        # Guard: Tavily occasionally returns non-LinkedIn URLs even
        # with include_domains. Skip anything off-platform.
        if "linkedin.com" not in url.lower():
            continue
        # Date resolution: prefer the LinkedIn snowflake-decoded
        # timestamp from the URL (deterministic, free) over Tavily's
        # metadata (often null or stale for LinkedIn). Fall back to
        # Tavily's published_date when the URL isn't a recognisable
        # activity / share / ugcPost.
        published_raw = r.get("published_date")
        url_dt = _decode_url_timestamp(url)
        if url_dt:
            published = url_dt.isoformat()
            date_source = "url_snowflake"
            if published_raw and published_raw != published:
                recovered_from_url += 1
        elif published_raw:
            published = published_raw
            date_source = "tavily"
        else:
            # No URL ID, no Tavily date — drop. Better than letting a
            # year-old post through as 'fresh' just because we can't
            # see its age.
            dropped_undated += 1
            continue
        if not _is_within_age(published):
            dropped_old += 1
            continue
        content = (r.get("content") or "").strip()
        items.append(
            {
                "source": f"linkedin:{tag}",
                "source_url": url,
                "title": title,
                "summary": content[:600],
                # intel_raw schema is strict — keep date inside raw_json.
                # tools/freshness.parse_published_at() picks it up from
                # raw_json.published_date during promotion.
                "raw_json": {
                    "tavily_score": r.get("score"),
                    "published_date": published,
                    "date_source": date_source,
                    "query_tag": tag,
                    "region": region,
                    "country_code": country_code,
                    # 2026-04-29: themed targets (proptech / materials / prime
                    # / capital-markets) ship category_hint so ingest routes
                    # them to the right page without classifier dependency.
                    **({"category_hint": theme} if theme else {}),
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
    log.info(
        "[linkedin:%s] query=%r kept=%d dropped_old=%d dropped_undated=%d url_recovered=%d",
        tag, query[:80], len(items), dropped_old, dropped_undated, recovered_from_url,
    )
    return items


# Per-query cache so MD's REFRESH clicks don't compound LinkedIn-Tavily
# cost. With pipeline at 2× daily (5 AM + 1 PM GST), each query gets
# called twice from the cron. A 4h TTL means manual REFRESH triggers
# inside the cron window all hit cache → zero added credits. (2026-04-30
# audit-round fix: 216 calls/day at 3× cron + REFRESH spam dropped to
# ~92 calls/day baseline.)
import time as _time
_LINKEDIN_QUERY_CACHE: dict[str, dict] = {}
_LINKEDIN_CACHE_TTL_SEC = 4 * 60 * 60  # 4 hours


async def _cached_query(tag: str, query: str, region: str, country_code: str,
                        theme: str | None) -> list[dict[str, Any]]:
    """Wrapper around _run_query with a per-query 4h cache."""
    cache_key = f"{tag}|{query}"
    cached = _LINKEDIN_QUERY_CACHE.get(cache_key)
    if cached and (_time.time() - cached.get("at", 0)) < _LINKEDIN_CACHE_TTL_SEC:
        return cached.get("rows") or []
    rows = await _run_query(tag, query, region=region,
                             country_code=country_code, theme=theme)
    _LINKEDIN_QUERY_CACHE[cache_key] = {"at": _time.time(), "rows": rows}
    return rows


async def run(limit: int = 40, force: bool = False) -> list[dict[str, Any]]:
    # 2026-05-11 stakeholder directive: LinkedIn scout runs ONCE per
    # day, in the morning slot. Tavily queries are the most expensive
    # source we have (~50 queries/cycle), so we gate by UTC hour:
    #   - 01:00 UTC = 05:00 GST (the daily morning pipeline) → run
    #   - any other firing time (13:00 / 21:00 GST, hourly keep-alive,
    #     manual /run/pipeline outside the window) → skip with a log
    # Override the gate one of three ways:
    #   (a) call run(force=True) directly (used by /run/pipeline?force_linkedin=1)
    #   (b) set env var LINKEDIN_FORCE=1 on Railway (persistent override)
    #   (c) wait for the morning slot
    import os as _os
    if not force and _os.environ.get("LINKEDIN_FORCE") != "1":
        from datetime import datetime as _dt, timezone as _tz
        utc_hour = _dt.now(_tz.utc).hour
        # Morning window: 00:00–04:59 UTC (covers 04:00–08:59 GST, so
        # the 01:00 UTC scheduled run lands squarely in the middle).
        if utc_hour > 4:
            log.info(
                "[linkedin] skipped — outside morning window "
                "(UTC hour %d). Pass force=True or set LINKEDIN_FORCE=1 to override.",
                utc_hour,
            )
            return []
    if force:
        log.info("[linkedin] force=True — bypassing morning gate")
    # (tag, query, region, country, theme)  — theme=None for legacy dev/broker queries.
    queries: list[tuple[str, str, str, str, str | None]] = []
    for dev_name, _slug, region, country in DEVS:
        qual = REGION_QUALIFIER.get(region, "")
        for tag, template in QUERY_TEMPLATES:
            q = template.format(dev=dev_name, qual=qual).replace("  ", " ").strip()
            queries.append((tag, q, region, country, None))
    # 2026-04-29: themed targets — one query per company with the broader
    # THEMED_TEMPLATE. Each is tagged with its theme so the resulting
    # signals route to the right themed page.
    for company, theme, region, country in THEMED_TARGETS:
        qual = REGION_QUALIFIER.get(region, "")
        q = THEMED_TEMPLATE.format(dev=company, qual=qual).replace("  ", " ").strip()
        queries.append((f"themed-{theme}", q, region, country, theme))

    batches = await asyncio.gather(
        *[_cached_query(tag, q, r, c, t) for tag, q, r, c, t in queries],
        return_exceptions=True,
    )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            log.warning("[linkedin] batch raised: %s", b)
            continue
        flat.extend(b)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        deduped.append(it)

    log.info("[linkedin] fetched=%d deduped=%d", len(flat), len(deduped))
    return deduped[:limit]
