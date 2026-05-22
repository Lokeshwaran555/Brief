"""Competitor expansion-footprint scout (Round-2, 2026-04-29).

Catches strategic moves that leak through OPERATIONAL channels weeks
before announcements:

  1. Domain / DNS: certificate-transparency log lookups via crt.sh
     surface new subdomains. emaar-saudiarabia.com appearing in CT logs
     precedes Emaar's KSA expansion announcement by months.
  2. App-store listings: when a developer pushes a new buyer app or
     updates an existing one, the changelog often signals new sales
     channels or markets.
  3. Patents / trademarks: UAE Ministry of Economy IP register +
     WIPO. Catches branding moves before launch.
  4. Trade-show exhibitor lists: Cityscape Global, IPS, BIG 5, MIPIM.
     Who's exhibiting where = where they're selling next.

Multi-source scout because the signals are sparse — bundling keeps the
pipeline footprint small while covering 4 leakage channels.

Stamp:
  - source: "competitor_footprint:<channel>"
  - category_hint: "competitor"
  - tier_hint:     "official" (CT logs / app-store / IP registers)
                   or "press" for trade-show coverage
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from tools import tavily_search

log = logging.getLogger(__name__)


# Top developers we monitor for footprint changes. Keep tight — every
# entry is one crt.sh lookup + a Tavily query, so cost grows linearly.
TARGET_DEVS: list[dict[str, str]] = [
    {"name": "Emaar",       "dev_slug": "emaar",     "domain": "emaar.com",            "region": "dubai"},
    {"name": "DAMAC",       "dev_slug": "damac",     "domain": "damacproperties.com",  "region": "dubai"},
    {"name": "Aldar",       "dev_slug": "aldar",     "domain": "aldar.com",            "region": "abu_dhabi"},
    {"name": "Nakheel",     "dev_slug": "nakheel",   "domain": "nakheel.com",          "region": "dubai"},
    {"name": "Modon",       "dev_slug": "modon",     "domain": "modon.ae",             "region": "abu_dhabi"},
    {"name": "Binghatti",   "dev_slug": "binghatti", "domain": "binghatti.com",        "region": "dubai"},
]


# ─── 1. crt.sh certificate-transparency lookup ───
# Surfaces every TLS cert issued for a domain — including subdomains.
# Free public JSON endpoint. Heavy on cold calls; we cache subdomain
# state across runs implicitly via the dedup_key logic.
CRT_URL = "https://crt.sh/?q={domain}&output=json"


async def _crt_subdomains(client: httpx.AsyncClient, dev: dict[str, str]) -> list[dict[str, Any]]:
    """Look for *new* subdomains under the dev's apex domain.
    A subdomain like 'saudiarabia.emaar.com' or 'kenya-launch.emaar.com'
    appearing in CT logs is a strong expansion signal.
    """
    try:
        resp = await client.get(
            CRT_URL.format(domain=f"%25.{dev['domain']}"),  # %25 = URL-encoded %
            headers={"User-Agent": "Sobha MDI competitor-footprint scout"},
            timeout=20.0,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        log.warning("[footprint:%s] crt.sh failed: %s", dev["dev_slug"], type(e).__name__)
        return []

    # Collect distinct subdomain names, filtered to interesting ones
    # (i.e. NOT generic www, mail, cdn, *.aws, etc.).
    seen_subs: set[str] = set()
    INTERESTING_RX = (
        "saudi", "ksa", "neom", "egypt", "india", "nigeria",
        "uk", "london", "us", "usa", "kenya", "qatar", "oman",
        "branded", "launch", "campaign", "hotel", "residences",
        "tower", "phase", "investor",
    )
    items: list[dict[str, Any]] = []
    for row in rows[:200]:
        name = (row.get("name_value") or "").lower().strip()
        if not name:
            continue
        for sub in name.split("\n"):
            sub = sub.strip()
            if not sub or sub == dev["domain"] or sub == f"www.{dev['domain']}":
                continue
            if sub in seen_subs:
                continue
            # Skip generic / infrastructural prefixes.
            if any(sub.startswith(g) for g in ("mail.", "smtp.", "imap.",
                                                "cdn.", "_dmarc.", "_acme",
                                                "autodiscover.", "lyncdiscover.",
                                                "sip.")):
                continue
            seen_subs.add(sub)
            # Only fire for subdomains that contain an interesting term
            # — otherwise we'd flood the brief with infrastructure noise.
            if not any(term in sub for term in INTERESTING_RX):
                continue
            issued = row.get("entry_timestamp") or row.get("not_before")
            # Only surface RECENT issuances (last 30d).
            try:
                ts = datetime.fromisoformat(str(issued).rstrip("Z"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - ts) > timedelta(days=30):
                    continue
            except Exception:
                continue
            title = f"{dev['name']} · new subdomain · {sub}"
            items.append({
                "source":     f"competitor_footprint:dns_{dev['dev_slug']}",
                "source_url": f"https://crt.sh/?q={sub}",
                "title":      title[:300],
                "summary":    (
                    f"Certificate-transparency log shows new subdomain '{sub}' "
                    f"under {dev['domain']} (issued {issued}). Subdomain naming "
                    f"often precedes geographic expansion or campaign launch."
                )[:600],
                "raw_json": {
                    "subdomain":   sub,
                    "issuer":      row.get("issuer_name"),
                    "issued":      issued,
                    "category_hint": "competitor",
                    "tier_hint":     "official",
                    "dev_slug_hint": dev["dev_slug"],
                    "region":         dev["region"],
                    "country_code":   "AE",
                    "channel":        "dns_ct_log",
                },
                "dedup_key": hashlib.sha256(
                    f"footprint:dns:{sub}".encode()
                ).hexdigest(),
            })
    return items


# ─── 2-4. Tavily-mediated app-store / IP register / trade-show queries ───
TAVILY_QUERIES: list[dict[str, Any]] = [
    {
        "channel": "app_store",
        "label":   "Mobile-app launch / update",
        "query":   "Emaar OR DAMAC OR Aldar OR Nakheel OR Sobha OR Binghatti new app launch update buyer 2026 iOS Android",
        "domains": ["apps.apple.com", "play.google.com",
                    "techcrunch.com", "thenationalnews.com",
                    "khaleejtimes.com", "constructionweekonline.com"],
    },
    {
        "channel": "ip_register",
        "label":   "Trademark / brand filing",
        "query":   "UAE Ministry Economy trademark Emaar OR DAMAC OR Aldar OR Sobha OR Binghatti 2026 brand filing",
        "domains": ["wipo.int", "moec.gov.ae", "ipsearch.gulfip.com",
                    "thenationalnews.com", "khaleejtimes.com"],
    },
    {
        "channel": "trade_show",
        "label":   "Trade-show exhibitor",
        "query":   "Cityscape Global IPS BIG 5 MIPIM 2026 exhibitor Emaar DAMAC Aldar Sobha Binghatti booth",
        "domains": ["cityscapeglobal.com", "ips.org", "thebig5.ae",
                    "mipim.com", "constructionweekonline.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "meed.com"],
    },
]


def _within_age(ts: Any, days: int = 30) -> bool:
    if not ts:
        return True
    try:
        s = str(ts).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=days)
    except Exception:
        return True


async def _fetch_tavily(channel: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        results = await tavily_search.search(
            channel["query"],
            max_results=4,
            topic="news",
            search_depth="basic",
            include_domains=channel["domains"],
            days=30,
        )
    except Exception as e:
        log.warning("[footprint:%s] tavily failed: %s", channel["channel"], e)
        return []
    items: list[dict[str, Any]] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()[:300]
        if not url or not title:
            continue
        if not _within_age(r.get("published_date")):
            continue
        head = (title or "").strip().lower()[:120]
        domain = urlparse(url).netloc.lower() if url else channel["channel"]
        dedup = hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()
        items.append({
            "source":     f"competitor_footprint:{channel['channel']}",
            "source_url": url,
            "title":      f"{channel['label']} · {title[:240]}",
            "summary":    (r.get("content") or "")[:600],
            "raw_json": {
                "channel":       channel["channel"],
                "published_date": r.get("published_date"),
                "tavily_score":  r.get("score"),
                "category_hint": "competitor",
                "tier_hint":     "official" if channel["channel"] in ("app_store", "ip_register") else "press",
                "region":         "dubai",
                "country_code":   "AE",
            },
            "dedup_key": dedup,
        })
    return items


async def run(limit: int = 16) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Channel 1 — crt.sh subdomain monitoring per dev.
    async with httpx.AsyncClient(follow_redirects=True) as client:
        crt_batches = await asyncio.gather(
            *[_crt_subdomains(client, dev) for dev in TARGET_DEVS],
            return_exceptions=True,
        )
        for b in crt_batches:
            if isinstance(b, Exception):
                continue
            out.extend(b)

    # Channels 2-4 — Tavily-mediated, parallel.
    tav_batches = await asyncio.gather(
        *[_fetch_tavily(c) for c in TAVILY_QUERIES],
        return_exceptions=True,
    )
    for b in tav_batches:
        if isinstance(b, Exception):
            continue
        out.extend(b)

    # Dedupe across channels.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in out:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        deduped.append(it)
    log.info("[competitor_footprint] devs=%d channels=%d total=%d deduped=%d",
             len(TARGET_DEVS), len(TAVILY_QUERIES), len(out), len(deduped))
    return deduped[:limit]
