"""Policy-wire scout — UAE government wire (WAM) + Zawya MENA wire.

Government policy announcements (Golden Visa expansion, freehold rule
changes, RERA amendments, DLD pronouncements) hit national wires hours
to days before mainstream press picks them up. This scout pulls the
firehose then prefilters to the five terms an MD actually cares about.

Five-term filter (per source audit §D.11): Golden Visa, freehold,
RERA, DLD, real estate. Anything not matching gets dropped at scout
time so we don't burn classifier tokens on irrelevant national news.

Sources:
  - WAM (Emirates News Agency) RSS — official UAE wire
  - Zawya MENA real estate wire — Refinitiv-owned, free RSS

Both are free with no key. We code defensively: any RSS URL drift
returns [] rather than poisoning the run.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from urllib.parse import quote as _urlquote

log = logging.getLogger(__name__)


def _gn(domain: str, keywords: str, *, gl: str = "AE", hl: str = "en", when: str = "14d") -> str:
    """Google News site-restricted RSS query. 2026-05-18 rewrite:
    publisher RSS endpoints dead (WAM returns HTML, Zawya 404).
    Hit Google's index instead."""
    q = f'site:{domain} ({keywords}) when:{when}'
    ceid = f"{gl}:{hl}"
    return f"https://news.google.com/rss/search?q={_urlquote(q)}&hl={hl}&gl={gl}&ceid={ceid}"


# Each entry: name, Google-News-site query, region/country tag.
# Keyword query is narrow — only policy / regulatory / RE moves.
_KW_POLICY = ('"Golden Visa" OR freehold OR RERA OR DLD OR Trakheesi OR '
              '"real estate" OR property OR mortgage OR "residency visa" OR '
              'Aldar OR Emaar OR DAMAC OR Mubadala OR sukuk OR REIT OR '
              '"central bank" OR escrow')

FEEDS: list[dict[str, str]] = [
    {
        "name": "WAM · UAE policy via GN",
        "url": _gn("wam.ae", _KW_POLICY),
        "region": "dubai",  # WAM covers UAE-wide; MD-scan defaults to dubai bucket
        "country_code": "AE",
    },
    {
        "name": "Zawya · UAE real estate via GN",
        "url": _gn("zawya.com", _KW_POLICY + ' (UAE OR Dubai OR "Abu Dhabi")'),
        "region": "dubai",
        "country_code": "AE",
    },
]


# Five-term prefilter — drops national-noise items at scout time.
# Case-insensitive substring match on title + summary.
KEEP_TERMS: tuple[str, ...] = (
    "golden visa",
    "freehold",
    "rera",
    "dld",
    "real estate",
    # Adjacent terms worth keeping — same regulatory family.
    "property law",
    "property visa",
    "investor visa",
    "ejari",
    "escrow",
    "mortgage cap",
    "ltv",
    "central bank",
    # Visa / residency vocabulary (added 2026-04-30 stakeholder ask:
    # MD wants visa-threshold changes — Golden Visa AED minimums,
    # property residency tiers — to surface promptly. Five-term
    # filter was too narrow; broaden to the residency vocabulary.)
    "visa threshold",
    "visa eligibility",
    "residency visa",
    "residence permit",
    "long-term residency",
    "long term residency",
    "10-year visa",
    "5-year visa",
    "retirement visa",
    "remote work visa",
    "digital nomad visa",
    "uae residency",
    "investor residency",
    "property residency",
    "naturalization",
    "citizenship",
    "iqama",
    # Infrastructure / megaproject vocabulary (added 2026-04-26).
    # Major transit + airport + smart-city announcements shift land
    # values; the Gold Line Metro extension was missed by prior filters.
    "infrastructure",
    "metro",
    "rail",
    "rta",
    "megaproject",
    "smart city",
    "airport",
    "port",
    "transit",
)


MAX_AGE_DAYS = 14


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "wire"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True
    try:
        s = str(published_str).rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


def _matches_filter(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in KEEP_TERMS)


async def _fetch(client: httpx.AsyncClient, feed: dict[str, str]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI policy-wire scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[policy_wire:%s] fetch failed: %s", feed["name"], e)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:30]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        summary = (getattr(e, "summary", "") or "").strip()
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        if not _matches_filter(title + " " + summary):
            continue
        out.append(
            {
                "source": f"policy_wire:{feed['name'].lower().replace(' ', '_')}",
                "source_url": link,
                "title": f"{feed['name'].split(' ')[0]} · {title[:240]}",
                "summary": summary[:600],
                "raw_json": {
                    "wire_source": feed["name"],
                    "published_date": published,
                    "region": feed["region"],
                    "country_code": feed["country_code"],
                    "category_hint": "policy",
                },
                "dedup_key": _dedup_key(title, link),
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
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch(client, feed) for feed in FEEDS],
            return_exceptions=False,
        )
    flat = [row for batch in batches for row in batch]
    deduped = _dedupe(flat)
    log.info(
        "[policy_wire] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
