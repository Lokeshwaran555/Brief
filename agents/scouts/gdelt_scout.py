"""GDELT 2.0 global news scout.

GDELT indexes ~150K news sources worldwide with country tagging
baked in — perfect for the CEO scanner's non-Dubai regions plus the
dynamic 'Others' bucket. Free, no API key, no rate-limit beyond
courtesy.

Doc API: https://api.gdeltproject.org/api/v2/doc/doc

We run one query per region bucket plus a global "everything else"
query. Country-code is preserved on each item so the dashboard can
auto-split 'Others' when 3+ items from the same country cluster.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any
from urllib.parse import quote, urlparse

import httpx

log = logging.getLogger(__name__)


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT uses full English country names in output. Map to ISO 3166-1
# alpha-2 for compact storage + dashboard grouping.
COUNTRY_NAME_TO_ISO = {
    "United States": "US",
    "Australia": "AU",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "India": "IN",
    "Saudi Arabia": "SA",
    "Singapore": "SG",
    "Canada": "CA",
    "Spain": "ES",
    "Italy": "IT",
    "Germany": "DE",
    "France": "FR",
    "Vietnam": "VN",
    "Thailand": "TH",
    "Indonesia": "ID",
    "Philippines": "PH",
    "Malaysia": "MY",
    "Hong Kong": "HK",
    "Japan": "JP",
    "China": "CN",
    "South Korea": "KR",
    "Egypt": "EG",
    "Brazil": "BR",
    "Mexico": "MX",
    "Poland": "PL",
    "Turkey": "TR",
    "South Africa": "ZA",
    "Kenya": "KE",
    "Nigeria": "NG",
    "Pakistan": "PK",
    "Bangladesh": "BD",
    "Sri Lanka": "LK",
    "Qatar": "QA",
    "Bahrain": "BH",
    "Kuwait": "KW",
    "Oman": "OM",
}


# RE-relevant theme combinations. GDELT supports complex query DSL
# with theme:, sourcecountry:, sourcelang: etc. We keep queries tight
# so each bucket returns on-topic articles.
QUERIES: list[dict[str, str]] = [
    # USA — Texas first, then nationwide multifamily / REIT activity.
    {
        "tag": "us-texas",
        "query": '("multifamily" OR "build-to-rent" OR "Class A apartments" OR "Texas real estate" OR "Austin development" OR "Dallas property") sourcecountry:US sourcelang:eng',
        "region": "usa",
        "country": "US",
    },
    {
        "tag": "us-reit-activity",
        "query": '("REIT" OR "8-K filing" OR "build-to-rent" OR "PropTech" OR "modular construction" OR "land acquisition") sourcecountry:US sourcelang:eng',
        "region": "usa",
        "country": "US",
    },
    # Australia — Brisbane priority, broader AU as fallback.
    {
        "tag": "au-brisbane",
        "query": '("Brisbane property" OR "Olympic infrastructure" OR "Queensland real estate" OR "luxury apartments") sourcecountry:AU sourcelang:eng',
        "region": "australia",
        "country": "AU",
    },
    {
        "tag": "au-developers",
        "query": '(Mirvac OR Stockland OR Lendlease OR "Goodman Group" OR "Charter Hall") sourcecountry:AU sourcelang:eng',
        "region": "australia",
        "country": "AU",
    },
    # UAE — Abu Dhabi first; Dubai is already covered by gnews/IG/Bayut/etc.
    {
        "tag": "ae-abu-dhabi",
        "query": '("Abu Dhabi real estate" OR Aldar OR Mubadala OR "Saadiyat Island" OR "Reem Island" OR "Yas Island") sourcecountry:AE sourcelang:eng',
        "region": "abu_dhabi",
        "country": "AE",
    },
    # Global / emerging — broad query, country code preserved per item.
    # Excludes the regions above so we don't double-ingest.
    {
        "tag": "global-emerging",
        "query": '("luxury residences" OR "branded residences" OR "real estate launch" OR "residential tower") -sourcecountry:US -sourcecountry:AE -sourcecountry:AU -sourcecountry:IN sourcelang:eng',
        "region": "other",
        "country": None,  # set per-item from sourcecountry
    },
    {
        "tag": "global-tech",
        "query": '("PropTech" OR "construction tech" OR "AI walkthrough" OR "modular pods" OR "prefab residential") -sourcecountry:US sourcelang:eng',
        "region": "other",
        "country": None,
    },
]


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "gdelt"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _country_iso(name: str | None) -> str | None:
    if not name:
        return None
    return COUNTRY_NAME_TO_ISO.get(name.strip())


def _gdelt_url(query: str, *, timespan: str = "7d", maxrecords: int = 50) -> str:
    return (
        f"{GDELT_DOC_API}?query={quote(query)}"
        f"&mode=ArtList&format=json&sort=DateDesc"
        f"&maxrecords={maxrecords}&timespan={timespan}"
    )


async def _fetch_one(client: httpx.AsyncClient, q: dict[str, str]) -> list[dict[str, Any]]:
    url = _gdelt_url(q["query"])
    try:
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (SobhaMDI/1.0 gdelt-scout)"},
            timeout=12.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[gdelt:%s] fetch failed: %s", q["tag"], e)
        return []

    articles = data.get("articles") or []
    items: list[dict[str, Any]] = []
    for a in articles:
        title = (a.get("title") or "").strip()[:300]
        url = (a.get("url") or "").strip()
        if not title or not url:
            continue
        country_iso = _country_iso(a.get("sourcecountry"))
        # Per-item country override for global queries; fixed for region buckets.
        country = q.get("country") or country_iso
        # GDELT seendate format: 20260425T093000Z. Convert to ISO.
        seen = a.get("seendate") or ""
        published_at = None
        if len(seen) >= 15:
            try:
                published_at = (
                    f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]}"
                    f"T{seen[9:11]}:{seen[11:13]}:{seen[13:15]}Z"
                )
            except Exception:
                pass
        items.append(
            {
                "source": f"gdelt:{q['tag']}",
                "source_url": url,
                "title": title,
                "summary": (a.get("excerpt") or "")[:600],
                # intel_raw schema is strict — scout-stamped region,
                # country, and publish-date all live inside raw_json.
                # Promotion (ingest_flow._promote_to_signals) reads
                # them back via tools/freshness.parse_published_at
                # for source_published_at, and direct raw_json lookup
                # for region / country_code.
                "raw_json": {
                    "query_tag": q["tag"],
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "sourcecountry_raw": a.get("sourcecountry"),
                    "country_iso": country_iso,
                    "seendate": seen,
                    "published_date": published_at,
                    "region": q["region"],
                    "country_code": country,
                },
                "dedup_key": _dedup_key(title, url),
            }
        )
    log.info(
        "[gdelt:%s] fetched=%d region=%s",
        q["tag"], len(items), q["region"],
    )
    return items


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
    """Scrape all GDELT region buckets in parallel, dedupe, return up to `limit`."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch_one(client, q) for q in QUERIES], return_exceptions=False
        )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info("[gdelt] total fetched=%d deduped=%d", len(flat), len(deduped))
    return deduped[:limit]
