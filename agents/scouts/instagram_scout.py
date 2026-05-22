"""Instagram scout via Apify (apify/instagram-scraper).

IG aggressively blocks raw scraping from cloud IPs (Railway included),
so direct httpx pulls die within a handful of requests. Apify proxies
through residential IPs and gives us a clean JSON response.

Cost: Apify charges ~$2.30 per 1k posts at the free-tier price. With
~25 profiles × 6 posts = 150 posts per run → ~$0.35/run. Daily runs fit
comfortably inside the $5/mo free credit.

The scout silently returns [] when APIFY_TOKEN is not set, so the rest
of the ingest pipeline never blocks on telemetry.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any
from urllib.parse import urlparse

from tools.apify_client import run_actor

log = logging.getLogger(__name__)


PROFILES: list[str] = [
    # ─── Developers — official ───
    "https://www.instagram.com/emaardubai/",
    "https://www.instagram.com/damacofficial/",
    "https://www.instagram.com/aldarproperties/",
    "https://www.instagram.com/nakheel_official/",
    "https://www.instagram.com/sobharealty/",
    "https://www.instagram.com/binghatti/",
    "https://www.instagram.com/azizidevelopments/",
    "https://www.instagram.com/omniyatofficial/",
    "https://www.instagram.com/ellingtonproperties/",
    "https://www.instagram.com/modonproperties/",
    "https://www.instagram.com/meraas/",
    "https://www.instagram.com/deyaaruae/",
    # ─── Brokers — break news/pricing often ahead of dev accounts ───
    "https://www.instagram.com/lewisallsopp/",
    "https://www.instagram.com/moeznrealestate/",
    "https://www.instagram.com/mattgreen.dubai/",
    "https://www.instagram.com/ahmadalzaabi/",
    "https://www.instagram.com/nadim.nasri/",
    "https://www.instagram.com/zach_amin/",
    "https://www.instagram.com/fam_properties/",
    "https://www.instagram.com/dubaisothebysrealty/",
    "https://www.instagram.com/driven_properties/",
    "https://www.instagram.com/luxhabitat/",
    "https://www.instagram.com/allsopp_allsopp/",
    "https://www.instagram.com/haus_and_haus/",
    "https://www.instagram.com/exclusivelinksrealestatebrokers/",
    # ─── Tech & AI in RE — proptech / contech / fractional / tokenized (2026-04-29) ───
    "https://www.instagram.com/stake_uae/",
    "https://www.instagram.com/smartcrowd/",
    "https://www.instagram.com/squareyards/",
    "https://www.instagram.com/propertyfinder/",
    "https://www.instagram.com/bayut/",
    # ─── Materials / construction supply — UAE suppliers (2026-04-29) ───
    "https://www.instagram.com/emiratesteel_official/",
    "https://www.instagram.com/lafargeholcim/",
    "https://www.instagram.com/conmix_uae/",
    "https://www.instagram.com/holcim/",
    # ─── Global RE prime + HNWI (2026-04-29) ───
    "https://www.instagram.com/knightfrank/",
    "https://www.instagram.com/savills/",
    "https://www.instagram.com/christiesinternationalrealestate/",
    "https://www.instagram.com/sothebysrealty/",
    "https://www.instagram.com/luxurylistings/",
    "https://www.instagram.com/henleypartners/",
    # ─── Capital markets — sukuk / banks publishing rate moves (2026-04-29) ───
    "https://www.instagram.com/fab_uae/",
    "https://www.instagram.com/emiratesnbd/",
]


# 2026-04-29: per-owner theme override. When an Instagram post comes
# from one of these handles, the scout stamps `category_hint` in
# raw_json so ingest_flow.upsert_signal routes it to the right themed
# page (Markets / Construction / Tech / Global) regardless of the
# classifier's bias toward legacy categories. Keys are lowercase IG
# usernames — match against post.ownerUsername.
OWNER_THEME: dict[str, str] = {
    # tech (proptech / contech / tokenized)
    "stake_uae":        "tech",
    "smartcrowd":       "tech",
    "squareyards":      "tech",
    "propertyfinder":   "tech",
    "bayut":            "tech",
    # materials / construction supply
    "emiratesteel_official": "materials",
    "lafargeholcim":    "materials",
    "conmix_uae":       "materials",
    "holcim":           "materials",
    # global_prime
    "knightfrank":      "global_prime",
    "savills":          "global_prime",
    "christiesinternationalrealestate": "global_prime",
    "sothebysrealty":   "global_prime",
    "luxurylistings":   "global_prime",
    "henleypartners":   "capital_flow",  # HNWI migration → capital flow page
    # capital_markets
    "fab_uae":          "capital_markets",
    "emiratesnbd":      "capital_markets",
}


RELEVANCE_RX = re.compile(
    r"(launch|launching|pre.?launch|eoi|expression of interest|"
    r"now selling|exclusive|limited|payment plan|post.?handover|"
    r"handover|delivery|ready|"
    r"dld|rera|sukuk|"
    r"psf|price|aed|million|billion|"
    r"emaar|damac|aldar|nakheel|meraas|binghatti|azizi|omniyat|ellington|"
    r"deyaar|modon|sobha|dubai holding|"
    r"reem island|palm jumeirah|downtown|business bay|marina|"
    r"hills|jvc|jvt|creek harbour|mbr city|meydan|dubailand|furjan|tilal|"
    # 2026-04-29 widen for themed-page sources:
    # proptech / contech / AI in RE
    r"proptech|contech|fractional|tokeniz|tokenis|smart home|"
    r"ai design|digital twin|bim|modular|3d.?print|prefab|"
    r"funding round|series [a-c]|seed|"
    # materials / construction supply
    r"rebar|steel price|cement|concrete|copper|aluminum|"
    r"shipping rate|lead time|hvac|elevator|generator|switchgear|"
    # prime / HNWI / capital markets
    r"prime|hnwi|wealth report|migration|residency|"
    r"yield|spread|cds|rating|treasury|eibor|mortgage rate)",
    re.IGNORECASE,
)


APIFY_ACTOR = "apify~instagram-scraper"


def _dedup_key(caption: str, post_url: str) -> str:
    domain = urlparse(post_url).netloc.lower() if post_url else "instagram"
    head = (caption or "").strip().lower()[:120]
    return hashlib.sha256(f"{head}|{domain}".encode()).hexdigest()


async def run(limit: int = 60) -> list[dict[str, Any]]:
    posts = await run_actor(
        APIFY_ACTOR,
        {
            "directUrls": PROFILES,
            "resultsType": "posts",
            "resultsLimit": 6,
            "searchType": "hashtag",
            "addParentData": False,
        },
        timeout_seconds=300,
    )
    if not posts:
        return []

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in posts:
        caption = (p.get("caption") or "").strip()
        post_url = (p.get("url") or "").strip()
        owner = p.get("ownerUsername") or p.get("owner_username") or ""
        if not caption or not post_url:
            continue
        if not RELEVANCE_RX.search(caption[:800]):
            continue
        key = _dedup_key(caption, post_url)
        if key in seen:
            continue
        seen.add(key)
        title = re.split(r"(?<=[.!?])\s|\n", caption.strip(), maxsplit=1)[0][:200]
        owner_lc = owner.lower()
        # 2026-04-29: per-owner theme override stamps category_hint so the
        # signal routes to the right themed page (Markets / Construction /
        # Tech / Global). Falls through to None for the existing developer
        # / broker accounts so the classifier still decides those.
        theme = OWNER_THEME.get(owner_lc)
        raw_json: dict[str, Any] = {
            "owner": owner,
            "timestamp": p.get("timestamp"),
            "likes": p.get("likesCount"),
            "comments": p.get("commentsCount"),
            "type": p.get("type"),
        }
        if theme:
            raw_json["category_hint"] = theme
        items.append(
            {
                "source": f"instagram:{owner_lc}" if owner else "instagram",
                "source_url": post_url,
                "title": title or caption[:200],
                "summary": caption[:600],
                "raw_json": raw_json,
                "dedup_key": key,
            }
        )

    log.info("[instagram] posts=%d kept=%d", len(posts), len(items))
    return items[:limit]
