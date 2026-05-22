"""Abu-Dhabi-specific press scout — free RSS sources only.

Closes the AD coverage gap surfaced in the 2026-05-11 stakeholder
review: only ~5 AD-tagged signals in the full 300-row /api/intel
slice. AD has structurally thinner editorial coverage than Dubai;
the existing scouts (uae_press, gnews-ad, policy_wire) covered some
of it but missed Abu Dhabi government / Aldar press / The National's
AD section / Property vertical.

This scout pulls AD-specific RSS endpoints DIRECTLY (no Tavily, no
Apify — all free). Each feed is tagged region=abu_dhabi at scout
time so promotion routes the resulting signals to the MD Scan
Abu Dhabi tab without waiting on the classifier.

Sources:
  - The National · Abu Dhabi section (AD-leaning daily; breaks gov news)
  - The National · Property vertical (Dubai + AD launches; we keep AD)
  - Khaleej Times · Abu Dhabi tag (when present)
  - Mediaoffice Abu Dhabi (government press releases)
  - Aldar press feed (when discoverable via RSS)
  - WAM English category feeds (Abu Dhabi-leaning gov wire)

Fail-soft: any 404 / network error returns [] for that source.
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
    """Google News site-restricted RSS query.

    2026-05-18 rewrite: every UAE publisher's own RSS feed is dead
    (12 returned 404, 4 returned 403 Cloudflare blocks, the rest
    returned HTML instead of XML). Google News still indexes these
    publishers reliably and exposes the index via its RSS endpoint,
    so we hit Google's RSS with a `site:<domain>` filter instead.

    `keywords` should narrow within the publisher (eg "Dubai
    property launch"). `when:Nd` is appended automatically.
    """
    q = f'site:{domain} ({keywords}) when:{when}'
    ceid = f"{gl}:{hl}"
    return f"https://news.google.com/rss/search?q={_urlquote(q)}&hl={hl}&gl={gl}&ceid={ceid}"


# Free RSS / Atom feeds with AD-leaning content. Each feed is tagged
# region=abu_dhabi unconditionally so the dashboard's AD tab fills
# up regardless of headline content (the keep-terms filter still
# drops generic non-RE noise). Verified candidate URLs as of
# 2026-05-11; defensive fetch — broken feeds return [].
# 2026-05-18 rewrite: all 16 publisher RSS feeds confirmed dead
# (404s on TheNational/Zawya/Bayut/TradeArabia, 403 Cloudflare on
# ArabianBusiness/Hotelier/ConstructionWeek, KT/Gulf News/WAM/MEED
# return HTML instead of XML). Replaced with Google News
# site-restricted RSS queries that hit Google's index instead of
# the publishers' broken endpoints. Feedparser handles GN RSS
# identically — KEEP_TERMS + region tagging unchanged.
#
# Keyword pattern per publisher is AD-skewed: city + sub-market +
# developer + RE moves vocabulary. Each query also constrains to
# `when:14d` so we catch genuinely fresh content.
_AD_KW = ('"Abu Dhabi" OR Aldar OR Modon OR Mubadala OR IHC OR '
          'Saadiyat OR "Yas Island" OR "Reem Island" OR Hudayriyat OR '
          'ADGM OR ADX OR Reportage OR "Eagle Hills" OR Tamouh OR Bloom')
_RE_KW = ('"real estate" OR property OR launch OR developer OR '
          '"off-plan" OR REIT OR sukuk OR "branded residence" OR '
          'mortgage OR Cityscape')

FEEDS: list[dict[str, str]] = [
    {"name": "The National · AD via GN",     "url": _gn("thenationalnews.com",   _AD_KW), "dev_slug_hint": None},
    {"name": "The National · Property",      "url": _gn("thenationalnews.com",   _RE_KW + ' Dubai OR "Abu Dhabi"'), "dev_slug_hint": None},
    {"name": "The National · Economy",       "url": _gn("thenationalnews.com",   '("Abu Dhabi" OR Aldar OR Mubadala OR IHC) (' + _RE_KW + ')'), "dev_slug_hint": None},
    {"name": "Khaleej Times · UAE",          "url": _gn("khaleejtimes.com",      _AD_KW + ' OR ' + _RE_KW), "dev_slug_hint": None},
    {"name": "Gulf News · Business",         "url": _gn("gulfnews.com",          _AD_KW + ' OR ' + _RE_KW), "dev_slug_hint": None},
    {"name": "WAM · Business",               "url": _gn("wam.ae",                _AD_KW + ' OR (UAE (' + _RE_KW + '))'), "dev_slug_hint": None},
    {"name": "TradeArabia · UAE",            "url": _gn("tradearabia.com",       _AD_KW + ' OR ' + _RE_KW), "dev_slug_hint": None},
    {"name": "Zawya · AD",                   "url": _gn("zawya.com",             _AD_KW), "dev_slug_hint": None},
    {"name": "Zawya · Real Estate",          "url": _gn("zawya.com",             _RE_KW + ' (Dubai OR "Abu Dhabi" OR UAE)'), "dev_slug_hint": None},
    {"name": "MEED · Real Estate",           "url": _gn("meed.com",              _RE_KW + ' Dubai OR "Abu Dhabi" OR UAE'), "dev_slug_hint": None},
    {"name": "MEED · Construction",          "url": _gn("meed.com",              'construction OR contractor OR EPC ("Abu Dhabi" OR UAE)'), "dev_slug_hint": None},
    {"name": "Construction Week ME",         "url": _gn("constructionweekonline.com", 'construction OR developer OR project ("Abu Dhabi" OR UAE OR Dubai)'), "dev_slug_hint": None},
    {"name": "Hotelier ME",                  "url": _gn("hoteliermiddleeast.com", 'hotel OR resort OR hospitality ("Abu Dhabi" OR Saadiyat OR "Yas Island" OR Dubai)'), "dev_slug_hint": None},
    {"name": "Arabian Business · RE",        "url": _gn("arabianbusiness.com",   _AD_KW + ' OR ' + _RE_KW), "dev_slug_hint": None},
]


# Keep-terms — same posture as uae_press_scout but skewed toward
# AD anchors. The filter catches: tracked AD developers, AD sub-
# markets, AD sovereign / capital actors, AD-specific regulatory /
# infrastructure terms, plus generic RE moves the classifier wants
# to see. We also keep mainstream UAE-wide RE/property terms so the
# AD section doesn't go empty when the feed is publishing
# UAE-general coverage.
KEEP_TERMS: tuple[str, ...] = (
    # AD-anchored developers + sovereign
    "aldar", "modon", "mubadala", "ihc", "international holding",
    "adq", "adia", "eagle hills", "reportage", "imkan", "bloom",
    "tamouh", "rak properties", "agility", "tabreed",
    # AD sub-markets & islands
    "saadiyat", "yas island", "yas bay", "yas acres", "reem island",
    "al maryah", "hudayriyat", "al jurf", "al raha", "al reem",
    "abu dhabi", "ad airport", "etihad", "louvre", "guggenheim",
    "adgm", "hub71", "twofour54", "khalifa city", "al ain",
    # AD regulatory / capital
    "adx", "dct abu dhabi", "department of culture", "dmt",
    "department of municipalities", "abu dhabi global market",
    # Generic RE moves (mirrors uae_press)
    "off-plan", "branded residence", "luxury",
    "real estate", "property launch", "ground breaking",
    "master plan", "phase 2", "phase 3", "tower", "villas",
    "sold out", "handover", "ipo", "sukuk", "reit",
    "joint venture", "acquisition", "partnership",
    # UAE-wide RE/policy that often anchors AD content
    "golden visa", "freehold", "residency visa", "investor visa",
    "developer", "developers",
)


MAX_AGE_DAYS = 7


def _dedup_key(title: str, url: str) -> str:
    domain = urlparse(url).netloc.lower() if url else "ad-press"
    head = (title or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


def _within_age(published_str: Any) -> bool:
    if not published_str:
        return True  # unknown date → keep, freshness layer decides
    try:
        s = str(published_str).rstrip("Z")
        try:
            dt = datetime.strptime(str(published_str), "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) <= timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return True


def _relevant(title: str, summary: str) -> bool:
    """True when the entry mentions at least one AD-anchored or
    RE-relevant term. Drops generic banking / sports / oil noise."""
    blob = ((title or "") + " " + (summary or "")).lower()
    return any(t in blob for t in KEEP_TERMS)


async def _fetch_feed(client: httpx.AsyncClient, feed: dict[str, str]) -> list[dict[str, Any]]:
    try:
        resp = await client.get(
            feed["url"],
            headers={"User-Agent": "Sobha MDI AD-press scout (contact: ops@sobha.com)"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.info("[ad_press:%s] fetch failed: %s", feed["name"], type(e).__name__)
        return []

    parsed = feedparser.parse(resp.text)
    out: list[dict[str, Any]] = []
    for e in parsed.entries[:25]:
        title = (getattr(e, "title", "") or "").strip()
        link  = (getattr(e, "link",  "") or "").strip()
        if not title or not link:
            continue
        summary = (getattr(e, "summary", "") or "").strip()
        if not _relevant(title, summary):
            continue
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        if not _within_age(published):
            continue
        # Strip embedded HTML tags from summary (RSS frequently leaks
        # <p> / <img> wrappers via feedparser).
        import re as _re
        summary_clean = _re.sub(r"<[^>]+>", "", summary)
        summary_clean = _re.sub(r"\s+", " ", summary_clean).strip()[:600]
        out.append(
            {
                "source":     f"ad_press:{feed['name'].split(' · ')[0].lower().replace(' ', '_')}",
                "source_url": link,
                "title":      title[:300],
                "summary":    summary_clean,
                "raw_json": {
                    "ad_press_source": feed["name"],
                    "published_date":  published,
                    # Hard-tag region=abu_dhabi at scout time.
                    # Promotion will skip the classifier inference path.
                    "region":          "abu_dhabi",
                    "country_code":    "AE",
                    "dev_slug_hint":   feed.get("dev_slug_hint"),
                },
                "dedup_key":  _dedup_key(title, link),
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


async def run(limit: int = 60) -> list[dict[str, Any]]:
    """Fetch every AD-leaning RSS in parallel, filter to relevant
    items, dedup across feeds, return up to `limit`.
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch_feed(client, f) for f in FEEDS],
            return_exceptions=True,
        )
    flat: list[dict[str, Any]] = []
    for b in batches:
        if isinstance(b, Exception):
            log.warning("[ad_press] batch raised: %s", b)
            continue
        flat.extend(b)
    deduped = _dedupe(flat)
    log.info(
        "[ad_press] feeds=%d fetched=%d deduped=%d",
        len(FEEDS), len(flat), len(deduped),
    )
    return deduped[:limit]
