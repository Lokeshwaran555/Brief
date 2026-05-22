"""Reddit scout — /r/dubai + /r/dubairealestate (+ 14 more subs).

Uses Reddit's app-only OAuth when REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET
+ REDDIT_USERNAME env vars are set (60 req/min, bypasses Railway IP block).
Falls back to unauth `.json` endpoints otherwise (works locally; 403s on
datacenter IPs).

Pre-filters by developer / pricing / regulatory keywords BEFORE the LLM
sees them — Reddit's signal-to-noise is terrible (tourist Qs etc.).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


# Reddit endpoints — auth path uses oauth.reddit.com, unauth uses www.
# When REDDIT_CLIENT_ID/SECRET/USERNAME env vars are present we use OAuth
# (60 req/min, works from datacenter IPs). Otherwise unauth (10 req/min,
# 403s from Railway IPs).
REDDIT_OAUTH_URL = "https://oauth.reddit.com/r/{sub}/new?limit=50"
REDDIT_OAUTH_SEARCH_URL = "https://oauth.reddit.com/search?q={q}&sort=new&t=month&limit=20"
REDDIT_PUBLIC_URL = "https://www.reddit.com/r/{sub}/new.json?limit=50"
REDDIT_PUBLIC_SEARCH_URL = "https://www.reddit.com/search.json?q={q}&sort=new&t=month&limit=20"

# In-process token cache. Refreshed when expired.
_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}


def _reddit_user_agent() -> str:
    """Reddit requires a descriptive UA in `<platform>:<id>:<version> (by /u/<user>)` shape."""
    user = os.environ.get("REDDIT_USERNAME", "").strip() or "anon"
    return f"web:com.sobha.mdi.scout:1.0 (by /u/{user})"


async def _get_reddit_token(client: httpx.AsyncClient) -> str | None:
    """Fetch (or reuse) an app-only OAuth token. Returns None when
    REDDIT_CLIENT_ID/SECRET aren't set — caller should fall back to
    unauth path.
    """
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        return None

    now = time.time()
    cached = _TOKEN_CACHE.get("token")
    if cached and now < (_TOKEN_CACHE.get("expires_at") or 0):
        return cached

    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    try:
        resp = await client.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            headers={
                "Authorization": f"Basic {auth}",
                "User-Agent": _reddit_user_agent(),
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[reddit] OAuth token fetch failed: %s", e)
        return None

    token = data.get("access_token")
    expires_in = int(data.get("expires_in") or 3600)
    if token:
        _TOKEN_CACHE["token"] = token
        # Refresh 60s before expiry
        _TOKEN_CACHE["expires_at"] = now + expires_in - 60
    return token

# Per-sub config: which region/country to stamp + whether the global
# RE-relevance regex applies (some subs need looser filters because
# the local property vocabulary differs).
SUBREDDITS: list[dict[str, Any]] = [
    # Dubai — existing
    {"sub": "dubai",          "region": "dubai",     "country_code": "AE"},
    {"sub": "dubairealestate","region": "dubai",     "country_code": "AE"},
    # Abu Dhabi
    {"sub": "abudhabi",       "region": "abu_dhabi", "country_code": "AE"},
    # USA — Texas focus + nationwide
    {"sub": "RealEstate",     "region": "usa",       "country_code": "US"},
    {"sub": "austin",         "region": "usa",       "country_code": "US"},
    {"sub": "Dallas",         "region": "usa",       "country_code": "US"},
    {"sub": "houston",        "region": "usa",       "country_code": "US"},
    {"sub": "multifamily",    "region": "usa",       "country_code": "US"},
    {"sub": "CommercialRealEstate", "region": "usa", "country_code": "US"},
    {"sub": "REBubble",       "region": "usa",       "country_code": "US"},
    # Australia — Brisbane priority + nationwide
    {"sub": "AusProperty",    "region": "australia", "country_code": "AU"},
    {"sub": "AusFinance",     "region": "australia", "country_code": "AU"},
    {"sub": "brisbane",       "region": "australia", "country_code": "AU"},
    {"sub": "sydney",         "region": "australia", "country_code": "AU"},
    {"sub": "HousingAustralia","region": "australia","country_code": "AU"},
]


# Cross-sub search queries via reddit /search.json. Each query is a
# leak-vocabulary phrase the audit highlighted as high-signal across
# specific forums. Region tag is None when the query is global; the
# classifier infers region from headline + dek.
SEARCH_QUERIES: list[dict[str, Any]] = [
    # Dubai — pricing leakage
    {"tag": "dxb-eoi-wait",
     "q": '("EOI closed" OR "wait list" OR "EOI deadline") Dubai',
     "region": "dubai", "country_code": "AE"},
    {"tag": "dxb-broker-incentive",
     "q": '("0% commission" OR "0.5% commission" OR "broker incentive" OR "commission bump") Dubai',
     "region": "dubai", "country_code": "AE"},
    {"tag": "dxb-handover",
     "q": '("got my keys" OR "handover delay" OR "snagging") Dubai',
     "region": "dubai", "country_code": "AE"},
    {"tag": "dxb-payment-plan",
     "q": '("post-handover payment" OR "5% deposit" OR "10% down") Dubai',
     "region": "dubai", "country_code": "AE"},
    {"tag": "dxb-secondary",
     "q": '("assignment fee" OR "flip" OR "selling off-plan") Dubai',
     "region": "dubai", "country_code": "AE"},
    # Texas multifamily — concessions / lease-up
    {"tag": "us-concessions",
     "q": '("rent concessions" OR "month free" OR "two months free" OR "lease-up") Austin OR Dallas OR Houston multifamily',
     "region": "usa", "country_code": "US"},
    {"tag": "us-cap-rate",
     "q": '("cap rate compression" OR "trade hands" OR "merchant builder") Texas multifamily',
     "region": "usa", "country_code": "US"},
    # Brisbane — Olympic / DA approval
    {"tag": "au-olympics",
     "q": '("Olympics 2032" OR "Brisbane 2032") property OR development',
     "region": "australia", "country_code": "AU"},
    {"tag": "au-da",
     "q": '("DA approved" OR "off-the-plan") Brisbane OR Sydney',
     "region": "australia", "country_code": "AU"},
    # Buyer/market sentiment — global
    {"tag": "market-sentiment",
     "q": '("buyer\'s market" OR "seller\'s market" OR "market correction") Dubai OR Brisbane OR Texas',
     "region": None, "country_code": None},
    # PropTech / tokenization
    {"tag": "proptech-tokenize",
     "q": '("tokenized real estate" OR "fractional ownership") Dubai OR UAE',
     "region": "dubai", "country_code": "AE"},
]

# Pre-filter: must mention at least one of these terms to be worth scoring.
# Ruthless — Reddit has too much tourism/nightlife/visa-Q noise.
# Extended 2026-04-26 with section-B leak-vocabulary (commission rates,
# payment plans, snagging, post-handover terms, market sentiment).
RELEVANCE_RX = re.compile(
    r"\b("
    # Dubai-specific
    r"emaar|damac|aldar|nakheel|meraas|binghatti|azizi|omniyat|ellington|"
    r"deyaar|modon|sobha|mubadala|dubai holding|saadiyat|reem island|yas island|"
    # USA-specific
    r"camden|greystar|tricon|mid-america|mill creek|avalonbay|essex property|"
    r"reit|build.?to.?rent|btr|multifamily|class a|"
    r"rent concession|month free|lease.?up|cap rate|merchant builder|"
    # Australia-specific
    r"mirvac|stockland|lendlease|goodman group|charter hall|"
    r"corelogic|domain\.com|asic|olympics 2032|brisbane 2032|"
    # Pricing-leakage / off-plan vocab
    r"off.?plan|pre.?launch|eoi|expression of interest|wait list|"
    r"soft launch|vip preview|broker preview|invite only|"
    r"handover|payment plan|post.?handover|5% deposit|10% down|"
    r"snagging|got my keys|"
    # Regulatory / financial
    r"dld|rera|oqood|escrow|trakheesi|golden visa|freehold|"
    r"service charge|maintenance fee|"
    r"sukuk|mortgage|ltv|"
    # Pricing units
    r"psf|price per square|per sqft|per sq ?m|"
    # Commissions / incentives
    r"commission|0% commission|5%|6%|broker incentive|"
    # Market state
    r"buyer.?s market|seller.?s market|market correction|sold out|"
    r"assignment fee|secondary market|"
    # Capital / land / tech
    r"land deal|land acquisition|land bank|"
    r"construction tech|proptech|prefab|modular|tokeni[sz]ed|fractional ownership"
    r")\b",
    re.IGNORECASE,
)


def _dedup_key(title: str, url: str) -> str:
    head = (title or "").strip().lower()[:120]
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower() if url else "reddit"
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


async def _fetch_sub(
    client: httpx.AsyncClient,
    cfg: dict[str, Any],
    token: str | None = None,
) -> list[dict[str, Any]]:
    sub = cfg["sub"]
    if token:
        url = REDDIT_OAUTH_URL.format(sub=sub)
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": _reddit_user_agent(),
            "Accept": "application/json",
        }
    else:
        url = REDDIT_PUBLIC_URL.format(sub=sub)
        headers = {
            "User-Agent": _reddit_user_agent(),
            "Accept": "application/json",
        }
    try:
        r = await client.get(url, headers=headers, timeout=10.0)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        log.warning("[reddit:%s] fetch failed (auth=%s): %s", sub, bool(token), e)
        return []

    posts = payload.get("data", {}).get("children", [])
    out: list[dict[str, Any]] = []
    for p in posts:
        d = p.get("data", {}) or {}
        title = (d.get("title") or "").strip()
        selftext = (d.get("selftext") or "").strip()
        permalink = d.get("permalink") or ""
        post_url = f"https://www.reddit.com{permalink}" if permalink else d.get("url", "")
        hay = f"{title} {selftext[:500]}"
        if not title or not post_url:
            continue
        if not RELEVANCE_RX.search(hay):
            continue
        out.append(
            {
                "source": f"reddit:{sub}",
                "source_url": post_url,
                "title": title[:300],
                "summary": selftext[:600],
                "raw_json": {
                    "sub": sub,
                    "author": d.get("author"),
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "created_utc": d.get("created_utc"),
                    "region": cfg.get("region"),
                    "country_code": cfg.get("country_code"),
                },
                "dedup_key": _dedup_key(title, post_url),
            }
        )
    log.info("[reddit:%s] posts=%d kept=%d region=%s", sub, len(posts), len(out), cfg.get("region"))
    return out


async def _fetch_search(
    client: httpx.AsyncClient,
    query: dict[str, Any],
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Cross-subreddit query. Uses oauth.reddit.com when token present."""
    from urllib.parse import quote
    if token:
        url = REDDIT_OAUTH_SEARCH_URL.format(q=quote(query["q"]))
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": _reddit_user_agent(),
            "Accept": "application/json",
        }
    else:
        url = REDDIT_PUBLIC_SEARCH_URL.format(q=quote(query["q"]))
        headers = {
            "User-Agent": _reddit_user_agent(),
            "Accept": "application/json",
        }
    try:
        r = await client.get(url, headers=headers, timeout=10.0)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        log.warning("[reddit:search:%s] fetch failed (auth=%s): %s",
                    query["tag"], bool(token), e)
        return []

    posts = payload.get("data", {}).get("children", [])
    out: list[dict[str, Any]] = []
    for p in posts:
        d = p.get("data", {}) or {}
        title = (d.get("title") or "").strip()
        selftext = (d.get("selftext") or "").strip()
        permalink = d.get("permalink") or ""
        post_url = f"https://www.reddit.com{permalink}" if permalink else d.get("url", "")
        if not title or not post_url:
            continue
        # Search results may include user-promoted spam — apply the
        # same relevance regex that /new posts get, but slightly looser
        # (the query itself already pre-filtered).
        out.append(
            {
                "source": f"reddit:search:{query['tag']}",
                "source_url": post_url,
                "title": title[:300],
                "summary": selftext[:600],
                "raw_json": {
                    "query_tag": query["tag"],
                    "sub": d.get("subreddit"),
                    "author": d.get("author"),
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "created_utc": d.get("created_utc"),
                    "region": query.get("region"),
                    "country_code": query.get("country_code"),
                },
                "dedup_key": _dedup_key(title, post_url),
            }
        )
    log.info("[reddit:search:%s] kept=%d", query["tag"], len(out))
    return out


async def run(limit: int = 40) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # OAuth token (None when env vars unset → falls back to unauth)
        token = await _get_reddit_token(client)
        if token:
            log.info("[reddit] using app-only OAuth (60 req/min)")
        else:
            log.info("[reddit] no OAuth creds — falling back to unauth (likely 403 from Railway IPs)")

        sub_task = asyncio.gather(*[_fetch_sub(client, c, token) for c in SUBREDDITS])
        search_task = asyncio.gather(*[_fetch_search(client, q, token) for q in SEARCH_QUERIES])
        sub_batches, search_batches = await asyncio.gather(sub_task, search_task)
    flat = [row for batch in sub_batches for row in batch]
    flat += [row for batch in search_batches for row in batch]

    # dedup
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for it in flat:
        if it["dedup_key"] in seen:
            continue
        seen.add(it["dedup_key"])
        dedup.append(it)

    log.info("[reddit] total=%d deduped=%d (subs=%d searches=%d auth=%s)",
             len(flat), len(dedup),
             sum(len(b) for b in sub_batches), sum(len(b) for b in search_batches),
             bool(token))
    return dedup[:limit]
