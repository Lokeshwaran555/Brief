"""SEC EDGAR 8-K scout — US-listed REIT material event filings.

8-K is filed when a public company has a material event to disclose
(M&A, capital raise, leadership change, ratings action). For US
multifamily REITs and adjacent operators it's often the FIRST public
signal of a strategic move — sometimes by hours, sometimes by weeks.

EDGAR is free, no API key, no rate limit beyond courtesy. Output is
ATOM XML keyed per company CIK (Central Index Key).

Tracks tier-1 US multifamily / RE companies the MD cares about. Each
maps to the same dev_slug as the LinkedIn scout so signals collapse
correctly in the radar.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import feedparser
import httpx

log = logging.getLogger(__name__)


# CIK = SEC's Central Index Key. Verified against EDGAR full-text search
# 2026-04-26. Format: zero-padded to 10 digits in URL.
COMPANIES: list[dict[str, str]] = [
    # ── Multifamily REITs (Sobha Texas + nationwide watch) ────
    {"name": "Camden Property Trust",      "cik": "0000906345", "slug": "camden",   "ticker": "CPT"},
    {"name": "Mid-America Apartment",      "cik": "0000912595", "slug": "maa",      "ticker": "MAA"},
    {"name": "AvalonBay Communities",      "cik": "0000915912", "slug": "avalonbay","ticker": "AVB"},
    {"name": "Equity Residential",         "cik": "0000906107", "slug": "eqr",      "ticker": "EQR"},
    {"name": "Essex Property Trust",       "cik": "0000920522", "slug": "ess",      "ticker": "ESS"},
    {"name": "UDR Inc",                    "cik": "0000074260", "slug": "udr",      "ticker": "UDR"},
    {"name": "Independence Realty Trust",  "cik": "0001466085", "slug": "irt",      "ticker": "IRT"},
    {"name": "Tricon Residential",         "cik": "0001861849", "slug": "tricon",   "ticker": "TCN"},
    # ── Diversified RE / capital — adjacent comps ──────────────
    {"name": "Howard Hughes Holdings",     "cik": "0001498828", "slug": "hhh",      "ticker": "HHH"},
    {"name": "Brookfield Property Partners","cik": "0001545772", "slug": "bpy",     "ticker": "BPY"},
]

# Filing types we actually care about. 8-K is the headline; the rest
# catch supplementary disclosures.
FILING_TYPES = ["8-K", "S-3", "424B", "DEF 14A"]

EDGAR_ATOM = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type={ftype}&dateb=&owner=include&count=10&output=atom"
)


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower() if url else "edgar"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


async def _fetch_company_filings(
    client: httpx.AsyncClient, company: dict[str, str], ftype: str,
) -> list[dict[str, Any]]:
    url = EDGAR_ATOM.format(cik=company["cik"], ftype=ftype)
    try:
        # SEC requires a descriptive User-Agent; generic UAs get 403'd.
        resp = await client.get(
            url,
            headers={
                "User-Agent": "Sobha MDI agents (contact: ops@sobha.com)",
                "Accept": "application/atom+xml,application/xml,text/xml",
            },
            timeout=12.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[edgar:%s/%s] fetch failed: %s", company["slug"], ftype, e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:5]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        # Title looks like '8-K - Camden Property Trust (0000906345) (Filer)'
        # — strip the boilerplate so the headline reads better.
        clean_title = title.split(" - ", 1)[0].strip() if " - " in title else title
        summary = (getattr(e, "summary", "") or "").strip()[:600]
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        out.append(
            {
                "source": f"sec_edgar:{ftype.lower()}",
                "source_url": link,
                "title": f"{company['ticker']} · {clean_title}: {company['name']}",
                "summary": summary,
                "raw_json": {
                    "cik": company["cik"],
                    "ticker": company["ticker"],
                    "filing_type": ftype,
                    "published_date": published,
                    "region": "usa",
                    "country_code": "US",
                    # dev_slug used downstream by ingest_flow's _dev_slug
                    # matcher — pre-tagging it here saves a regex lookup
                    # and avoids collision with similarly-named non-RE
                    # companies in the global signal pool.
                    "dev_slug_hint": company["slug"],
                },
                "dedup_key": _dedup_key(clean_title + " " + company["ticker"], link),
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


async def run(limit: int = 40) -> list[dict[str, Any]]:
    """Pull recent SEC EDGAR filings for tracked companies. Each company
    × filing type fires in parallel; failures isolated per fetch.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        coros = []
        for company in COMPANIES:
            for ftype in FILING_TYPES:
                coros.append(_fetch_company_filings(client, company, ftype))
        batches = await asyncio.gather(*coros, return_exceptions=False)

    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info("[sec_edgar] fetched=%d deduped=%d (across %d companies × %d filing types)",
             len(flat), len(deduped), len(COMPANIES), len(FILING_TYPES))
    return deduped[:limit]
