"""DLD eMart distressed-auctions scout — luxury distressed inventory.

emart.dubailand.gov.ae is the DLD's official platform for property
auctions and distressed-asset sales in Dubai. When a luxury unit goes
to auction it appears here weeks before broader market notices —
material for distress-tracking + opportunistic acquisition signals.

Strategy: scrape the Featured Listings repeater on the home page
(ASP.NET WebForms — listings rendered server-side into a table). Each
listing has a detail-page link, photo, and basic metadata. We extract
the URL, title text, and any visible price / area / type, then stamp
region=dubai.

Free, no auth (the home page is fully public). Fail-soft: any HTML
shape change returns []. Light enough to run every ingest cycle —
single page fetch, ~50KB.

Detail-page deep dives are deliberately deferred — the home-page
listings carry enough metadata to seed a signal, and the Listing.aspx
detail pages add modest value at the cost of N extra requests per run.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any
from urllib.parse import urljoin

import httpx

log = logging.getLogger(__name__)


EMART_URL = "https://emart.dubailand.gov.ae/Home.aspx"

# Featured listings live inside <div class="mosaic-block ...">. Each
# block contains an <a href="Listings/ListingAuctionDetails.aspx?..."
# id="...lnkListingDetails"> wrapping a <h4> title and <p class="price">.
# Verified live 2026-04-26 against the production HTML.
LISTING_BLOCK_RE = re.compile(
    r'<a\s+href="(Listings/ListingAuctionDetails\.aspx\?[^"]+)"\s+'
    r'id="ctl00_PageContent_ucFeaturedListings_rptFeaturedListings_ctl(\d+)_lnkListingDetails"'
    r'[^>]*>(.*?)</a>',
    re.DOTALL,
)
TITLE_RE = re.compile(r"<h4[^>]*>(.*?)</h4>", re.DOTALL)
PRICE_RE = re.compile(r'<p[^>]*class="price"[^>]*>(.*?)</p>', re.DOTALL)

TAG_STRIP_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def _dedup_key(listing_url: str) -> str:
    return hashlib.sha256(f"emart:{listing_url}".encode()).hexdigest()


def _clean(text: str) -> str:
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


async def run(limit: int = 12) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                EMART_URL,
                headers={
                    "User-Agent": "Sobha MDI eMart-auctions scout (contact: ops@sobha.com)",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
    except Exception as e:
        log.warning("[emart] home fetch failed: %s", e)
        return []

    html = resp.text
    matches = LISTING_BLOCK_RE.findall(html)
    if not matches:
        log.warning("[emart] no listings matched repeater regex — HTML may have shifted")
        return []

    out: list[dict[str, Any]] = []
    for href, ctl_idx, inner_html in matches:
        listing_url = urljoin(EMART_URL, href)

        title_m = TITLE_RE.search(inner_html)
        title_text = _clean(title_m.group(1)) if title_m else ""

        price_m = PRICE_RE.search(inner_html)
        price = _clean(price_m.group(1)) if price_m else None

        # If <h4> empty (image-only listings exist), fall back to the
        # detail-page URL slug for a stable headline.
        if not title_text:
            title_text = listing_url.rsplit("/", 1)[-1][:80]

        headline_bits = ["DLD eMart auction"]
        if title_text:
            headline_bits.append(f"— {title_text}")
        if price:
            headline_bits.append(f"[{price}]")
        headline = " ".join(headline_bits)[:300]

        summary_bits = [f"DLD eMart featured auction listing: {title_text}"]
        if price:
            summary_bits.append(f"Price: {price}")
        summary_bits.append(f"Detail page: {listing_url}")
        summary = " · ".join(summary_bits)[:600]

        out.append(
            {
                "source": "emart:auction",
                "source_url": listing_url,
                "title": headline,
                "summary": summary,
                "raw_json": {
                    "listing_text": title_text,
                    "price_text": price,
                    "ctl_index": ctl_idx,
                    "region": "dubai",
                    "country_code": "AE",
                    "category_hint": "distressed_auction",
                },
                "dedup_key": _dedup_key(listing_url),
            }
        )

    # Dedupe within the page (defensive — the repeater shouldn't dupe).
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in out:
        k = it["dedup_key"]
        if k in seen:
            continue
        seen.add(k)
        deduped.append(it)

    log.info("[emart] listings=%d deduped=%d", len(out), len(deduped))
    return deduped[:limit]
