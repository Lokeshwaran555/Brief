"""FastAPI entrypoint.

Endpoints:
  GET  /health                     liveness probe
  GET  /status                     scheduler state + env sanity
  POST /run/ingest                 manually trigger ingest_flow
  POST /run/daily-brief            manually trigger daily brief (A+C)
  POST /deep-dive/{dev_slug}       on-demand per-developer crew (B)
  GET  /daily-brief                latest daily brief (or ?day=YYYY-MM-DD)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import json as _json

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import schedule as sched
from crews import deep_dive_crew
from flows import ceo_scan_flow, daily_brief_flow, ingest_flow, investigation_flow
from settings import settings
from tools import azure_intel, observability, sobha_context, supabase_tool as sb

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("server")


VALID_DEVS = {
    "emaar", "damac", "aldar", "nakheel", "modon", "dubaih",
    "binghatti", "azizi", "ellington", "omniyat", "deyaar",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    observability.init()
    sched.start()
    try:
        yield
    finally:
        sched.shutdown()


app = FastAPI(title="Sobha MDI Agents", lifespan=lifespan)

# 2026-04-30 audit-round: dropped localhost origins from production.
# Previously any rogue local server (e.g. another app the MD has open)
# could send writes to /api/watchlist /api/telemetry /run/pipeline.
# Local dev is gated by the ENV check below — only adds localhost
# origins when ENV=development is explicitly set.
ALLOWED_ORIGINS: list[str] = ["https://sobha-mdi.vercel.app"]
if os.environ.get("ENV", "").lower() == "development":
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000", "http://localhost:5173",
        "http://localhost:8000", "http://localhost:8090",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173",
        "http://127.0.0.1:8000", "http://127.0.0.1:8090",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # 2026-04-29: DELETE added for /api/watchlist/{id} (un-pin).
    # PATCH/PUT held back until a real use case lands.
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Audit-round 2026-04-30: shared admin auth gate. ADMIN_SECRET env var
# guards every /admin/* endpoint that returns DB content. When ENV=
# development AND ADMIN_SECRET is unset, the gate falls open so local
# dev still works. In production (Railway) ADMIN_SECRET MUST be set.
def _require_admin(authorization: str | None) -> None:
    expected = os.environ.get("ADMIN_SECRET", "").strip()
    if not expected:
        if os.environ.get("ENV", "").lower() == "development":
            return
        raise HTTPException(
            status_code=503,
            detail="ADMIN_SECRET env var not configured on this deploy",
        )
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid auth")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    jobs = [
        {
            "id": j.id,
            "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
        }
        for j in sched.scheduler.get_jobs()
    ]
    db = {
        "latest_signal_at": None,
        "latest_brief_day": None,
        "signals_active": None,
        "raw_pending": None,
    }
    try:
        latest_signal = (
            sb.client()
            .table("signals")
            .select("last_seen_at")
            .eq("archived", False)
            .order("last_seen_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        latest_brief = (
            sb.client()
            .table("daily_briefs")
            .select("day,generated_at")
            .order("day", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        active = (
            sb.client()
            .table("signals")
            .select("id", count="exact")
            .eq("archived", False)
            .limit(1)
            .execute()
        )
        pending = (
            sb.client()
            .table("intel_raw")
            .select("id", count="exact")
            .eq("processed", False)
            .limit(1)
            .execute()
        )
        db.update(
            {
                "latest_signal_at": latest_signal[0]["last_seen_at"] if latest_signal else None,
                "latest_brief_day": latest_brief[0]["day"] if latest_brief else None,
                "latest_brief_generated_at": latest_brief[0]["generated_at"] if latest_brief else None,
                "signals_active": active.count,
                "raw_pending": pending.count,
            }
        )
    except Exception as e:
        log.warning("status db summary failed: %s", e)
        db["error"] = str(e)

    # 2026-04-30 audit-round: report all three model identities, not
    # just nvidia (which was misleading — the chat path uses Groq).
    return {
        "scheduler_running": sched.scheduler.running,
        "jobs": jobs,
        "ingest_interval_minutes": settings.ingest_interval_minutes,
        "models": {
            "chat":    getattr(settings, "llm_model", None),
            "embed":   getattr(settings, "nvidia_model", None),
            "crew":    getattr(settings, "crew_model", None),
        },
        # Legacy: kept for back-compat with status pollers reading "model".
        "model": getattr(settings, "llm_model", settings.nvidia_model),
        "scouts": {
            "gnews": True,
            "reddit": True,
            "youtube": True,
            "forum": True,
            "instagram": bool(settings.apify_token),
        },
        "db": db,
    }


@app.get("/debug/friday-context")
async def debug_friday_context() -> dict:
    """Return the actual context blocks Friday assembles for a brief
    question, without calling the LLM. Used to diagnose hallucination:
    if a block is empty, we know Friday is making content up rather
    than missing data."""
    sobha_summary = sobha_context.format_for_llm("summary")
    market_block = ""
    try:
        from tools import market_snapshot as _ms
        snap = await _ms.get_snapshot()
        market_block = snap.get("one_liner") or ""
    except Exception:
        pass
    agenda_block = await _today_agenda_block()
    headlines_block = await _today_headlines_block(limit=10)
    brief_block = await _today_brief_block()
    return {
        "sobha_summary_len":   len(sobha_summary),
        "market_block":        market_block,
        "agenda_block":        agenda_block or "(empty)",
        "headlines_block":     headlines_block or "(empty)",
        "brief_block_len":     len(brief_block),
        "brief_block_preview": (brief_block or "(empty)")[:2000],
    }


@app.post("/admin/seed-leader-linkedin")
async def admin_seed_leader_linkedin(cleanup: int = 0, debug: int = 0) -> dict:
    """Fetch the latest real LinkedIn post per tracked leader via
    Apify and upsert as `signals` rows. Replaces the dashboard's
    `demo:leader_feed` rows with real content for the Peer Pulse
    tab. Idempotent: re-running updates the existing rows in place.

    cleanup=1 deletes any rows previously created by this endpoint.

    Cost: ~$0.05-0.10 per full run at $5/1k posts.

    Actor: tries a known-public LinkedIn-profile-posts actor first
    and falls back to alternates if it's deprecated. Failed actor
    runs return empty → that leader's slot stays unfilled rather
    than crashing the whole call.
    """
    from tools.apify_client import run_actor
    from tools import supabase_tool as _sb
    import hashlib as _hl
    from datetime import datetime as _dt, timezone as _tz

    # Curated leader URLs. Personal LinkedIn profile preferred where
    # the leader has one and posts regularly; company page otherwise.
    LEADERS_LI = [
        # ─── Dubai ───────────────────────────────────────────────
        {"name": "Hussain Sajwani",      "dev_slug": "damac",     "region": "dubai",
         "url": "https://www.linkedin.com/in/hussainsajwani"},
        {"name": "Mohamed Alabbar",      "dev_slug": "emaar",     "region": "dubai",
         "url": "https://www.linkedin.com/in/mohamed-alabbar-71b1aa108"},
        {"name": "DAMAC Properties",     "dev_slug": "damac",     "region": "dubai",
         "url": "https://www.linkedin.com/company/damacproperties"},
        {"name": "Emaar Properties",     "dev_slug": "emaar",     "region": "dubai",
         "url": "https://www.linkedin.com/company/emaar-properties"},
        {"name": "Binghatti Developers", "dev_slug": "binghatti", "region": "dubai",
         "url": "https://www.linkedin.com/company/binghatti-developers"},
        {"name": "Azizi Developments",   "dev_slug": "azizi",     "region": "dubai",
         "url": "https://www.linkedin.com/company/azizi-developments"},
        {"name": "Ellington Properties", "dev_slug": "ellington", "region": "dubai",
         "url": "https://www.linkedin.com/company/ellington-properties"},
        {"name": "Omniyat",              "dev_slug": "omniyat",   "region": "dubai",
         "url": "https://www.linkedin.com/company/omniyat"},
        {"name": "Nakheel",              "dev_slug": "nakheel",   "region": "dubai",
         "url": "https://www.linkedin.com/company/nakheel"},
        # ─── Abu Dhabi ───────────────────────────────────────────
        {"name": "Aldar Properties",     "dev_slug": "aldar",     "region": "abu_dhabi",
         "url": "https://www.linkedin.com/company/aldar-properties"},
        {"name": "Modon Properties",     "dev_slug": "modon",     "region": "abu_dhabi",
         "url": "https://www.linkedin.com/company/modon-properties"},
        {"name": "Mubadala",             "dev_slug": "mubadala",  "region": "abu_dhabi",
         "url": "https://www.linkedin.com/company/mubadala"},
        {"name": "IHC",                  "dev_slug": "ihc",       "region": "abu_dhabi",
         "url": "https://www.linkedin.com/company/international-holding-company"},
        # ─── Australia ───────────────────────────────────────────
        {"name": "Lendlease",            "dev_slug": "lendlease", "region": "australia",
         "url": "https://www.linkedin.com/company/lendlease"},
        {"name": "Mirvac",               "dev_slug": "mirvac",    "region": "australia",
         "url": "https://www.linkedin.com/company/mirvac"},
        {"name": "Stockland",            "dev_slug": "stockland", "region": "australia",
         "url": "https://www.linkedin.com/company/stockland"},
    ]

    def _cluster_key(name: str, profile_url: str) -> str:
        """Stable cluster_key so re-runs upsert in place."""
        stem = f"linkedin-peer-pulse|{name.lower()}|{profile_url.lower()}"
        return _hl.sha256(stem.encode()).hexdigest()

    sb = _sb.client()

    if cleanup:
        keys = [_cluster_key(L["name"], L["url"]) for L in LEADERS_LI]
        resp = sb.table("signals").delete(count="exact").in_("cluster_key", keys).execute()
        return {"ok": True, "action": "cleanup", "deleted": getattr(resp, "count", None)}

    # Apify actor — uses a public LinkedIn-profile-posts scraper.
    # Fallback list in order of preference; the first one that returns
    # data wins. If all fail, the function returns a clear status.
    ACTOR_CANDIDATES = [
        "apimaestro~linkedin-profile-posts",
        "harvestapi~linkedin-profile-posts",
        "apify~linkedin-profile-scraper",
        "curious_coder~linkedin-post-search-scraper",
    ]

    results: list[dict] = []
    actor_used = None
    for actor in ACTOR_CANDIDATES:
        run_input = {
            "profileUrls": [L["url"] for L in LEADERS_LI],
            "urls":        [L["url"] for L in LEADERS_LI],  # some actors use 'urls'
            "maxPosts":    1,
            "maxItems":    len(LEADERS_LI),
            "proxyConfiguration": {"useApifyProxy": True},
        }
        items = await run_actor(actor, run_input, timeout_seconds=180)
        if items:
            actor_used = actor
            results = items
            break

    if not results:
        return {
            "ok": False,
            "error": "All Apify LinkedIn actors returned empty",
            "actors_tried": ACTOR_CANDIDATES,
            "hint": "Check the Apify dashboard for currently-available LinkedIn post actors. The actor name may have changed; update ACTOR_CANDIDATES in /admin/seed-leader-linkedin.",
        }

    if debug:
        # Return first 2 raw items + key list so we can see the
        # actor's schema and adjust matching.
        sample = []
        for r in results[:2]:
            keys = list(r.keys()) if isinstance(r, dict) else []
            sample.append({"keys": keys, "first200chars": str(r)[:600]})
        return {
            "ok": True,
            "debug": True,
            "actor_used": actor_used,
            "raw_count": len(results),
            "sample_items": sample,
        }

    # Match Apify results back to leaders. Actor schemas vary
    # wildly — try every URL field, every author-name field, AND
    # do a substring sniff on the post text as a last resort.
    def _all_strings(obj, depth=0):
        if depth > 4: return
        if isinstance(obj, str):
            yield obj.lower()
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _all_strings(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj[:10]:
                yield from _all_strings(v, depth + 1)

    def _post_haystack(r):
        # Concatenate every string in the post for substring matching.
        return " ".join(_all_strings(r))

    def _post_url(r):
        # Try every known field name across the popular LinkedIn actors.
        for k in ("postUrl", "url", "post_url", "share_url", "shareUrl",
                  "permalink", "urn", "postLink", "link"):
            v = r.get(k)
            if isinstance(v, str) and "linkedin.com" in v:
                return v
        return None

    def _post_text(r):
        for k in ("text", "content", "postContent", "post_text", "description",
                  "caption", "commentary"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _post_date(r):
        for k in ("publishedAt", "postedAt", "date", "postedDate",
                  "publishedTime", "createdAt", "time"):
            v = r.get(k)
            if v:
                return str(v)
        return None

    rows = []
    used_results = set()
    now_iso = _dt.now(_tz.utc).isoformat()
    for L in LEADERS_LI:
        leader_url = L["url"].lower().rstrip("/")
        # Pull the LinkedIn handle from the URL — used for haystack
        # match when the URL field itself is missing.
        handle = leader_url.rstrip("/").split("/")[-1]
        leader_name_lc = L["name"].lower()
        post = None
        for idx, r in enumerate(results):
            if idx in used_results:
                continue
            hay = _post_haystack(r)
            # Match if the leader URL, the handle, or the name appears
            # anywhere in the post payload.
            if (leader_url in hay) or (handle and handle in hay) or (leader_name_lc in hay):
                post = r
                used_results.add(idx)
                break
        if not post:
            continue
        # Extract post URL + content via the per-shape helpers above.
        post_url = _post_url(post) or L["url"]
        text = _post_text(post)
        if not text:
            continue
        headline = (text[:200] + ("…" if len(text) > 200 else "")).replace("\n", " ")
        dek = text[:600] if len(text) > 200 else ""
        published_at = _post_date(post) or now_iso
        rows.append({
            "cluster_key": _cluster_key(L["name"], L["url"]),
            "first_seen_at": now_iso,
            "last_seen_at": now_iso,
            "source_count": 1,
            "dev_slug": L["dev_slug"],
            "category": "competitor",
            "decision_tag": "launch",
            "priority": "medium",
            "confidence": 0.85,
            "headline": f"{L['name']} · {headline}",
            "dek": dek,
            "entities": [L["name"]],
            "urls": [post_url],
            "client_score": 0.75,
            "archived": False,
            "region": L["region"],
            "country_code": "AE" if L["region"] in ("dubai", "abu_dhabi") else "AU",
            "source_published_at": published_at,
        })

    if not rows:
        return {
            "ok": False,
            "error": "Actor returned data but no posts mapped to leader URLs",
            "actor_used": actor_used,
            "raw_count": len(results),
        }

    # Upsert via delete + insert (signals.cluster_key is unique).
    keys = [r["cluster_key"] for r in rows]
    try:
        sb.table("signals").delete().in_("cluster_key", keys).execute()
    except Exception:
        pass
    # Insert; on schema-drift errors, retry without optional cols.
    try:
        ins = sb.table("signals").insert(rows).execute()
        inserted = len(ins.data or [])
    except Exception as e:
        log.warning("seed-leader-linkedin: full insert failed (%s), retrying without optional cols", e)
        min_rows = [{k: v for k, v in r.items() if k not in ("region", "country_code", "source_published_at")} for r in rows]
        ins = sb.table("signals").insert(min_rows).execute()
        inserted = len(ins.data or [])
    return {
        "ok": True,
        "action": "seed",
        "actor_used": actor_used,
        "inserted": inserted,
        "leaders_with_posts": len(rows),
        "leaders_total": len(LEADERS_LI),
    }


@app.get("/debug/linkedin-probe")
async def debug_linkedin_probe() -> dict:
    """Fire linkedin_scout directly (bypassing the gate) and report
    what came back. Diagnostic for "Leader Feed is empty" — surfaces
    whether the issue is upstream (Tavily key / empty results) or
    downstream (classifier rejected everything).
    """
    from agents.scouts import linkedin_scout as _li
    try:
        rows = await _li.run(limit=20, force=True)
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)[:300]}
    from settings import settings as _settings_probe
    return {
        "count": len(rows),
        "tavily_key_set": bool(_settings_probe.tavily_api_key),
        "tavily_key_prefix": (_settings_probe.tavily_api_key[:6] + "…") if _settings_probe.tavily_api_key else None,
        "sample": [
            {
                "title": (r.get("title") or "")[:120],
                "url":   (r.get("source_url") or "")[:200],
                "region": (r.get("raw_json") or {}).get("region"),
            }
            for r in rows[:5]
        ],
    }


@app.get("/debug/scout-counts")
async def debug_scout_counts(only: str | None = None, timeout_s: int = 60) -> dict:
    """Run each scout once and report raw item counts BEFORE
    classification / dedup / promotion. Diagnostic for "no
    Dubai/AD news" type complaints — tells us which specific
    scout returned [] vs which actually fetched items.

    Query params:
      ?only=gnews,uae_press   comma-separated subset (faster probe)
      ?timeout_s=60           per-scout timeout (default 60s)

    Returns:
      {
        "ran_at": ISO,
        "duration_total_s": float,
        "scouts": {
          "<scout_name>": {
            "raw_count": int,         # items returned by scout.run()
            "duration_s": float,
            "regions": {region: count, ...},  # raw_json.region tally
            "first_titles": [str, ...],       # 3 examples for sanity
            "error": str | None
          }, ...
        },
        "summary": {
          "scouts_with_items": int,
          "scouts_zero": int,
          "scouts_errored": int,
          "by_region": {region: count, ...}
        }
      }

    No DB writes. Each scout fires independently with its own try/
    except so one broken scout doesn't poison the report.
    """
    import asyncio as _asyncio
    import time as _time
    from agents.scouts import (
        ad_press_scout, architecture_scout, au_building_scout,
        au_press_scout, bdi_scfi_scout, cbuae_circulars_scout,
        cme_fedwatch_scout, competitor_footprint_scout,
        dfm_disclosures_scout, dld_projects_scout,
        eibor_scout, em_real_estate_scout, emart_auctions_scout,
        forum_scout, gdelt_scout, geopolitics_scout,
        github_leaks_scout, global_hpi_scout, global_prime_scout,
        gnews_scout, gtrends_scout, houston_permits_scout,
        industry_research_scout, ir_pages_scout, linkedin_scout,
        lme_metals_scout, macro_scout, materials_scout,
        nasdaq_dubai_scout, newsletters_scout, page_hash_scout,
        policy_wire_scout, proptech_releasenotes_scout,
        proptech_scout, proptech_vc_scout, reddit_scout,
        sanctions_scout, sec_edgar_scout, sukuk_scout,
        sydney_auctions_scout, trade_press_scout, trakheesi_scout,
        uae_cement_scout, uae_press_scout, us_permits_scout,
        wayback_diff_scout, youtube_scout,
    )

    SCOUTS = {
        "ad_press": ad_press_scout,
        "architecture": architecture_scout,
        "au_building": au_building_scout,
        "au_press": au_press_scout,
        "bdi_scfi": bdi_scfi_scout,
        "cbuae_circulars": cbuae_circulars_scout,
        "cme_fedwatch": cme_fedwatch_scout,
        "competitor_footprint": competitor_footprint_scout,
        "dfm_disclosures": dfm_disclosures_scout,
        "dld_projects": dld_projects_scout,
        "eibor": eibor_scout,
        "em_real_estate": em_real_estate_scout,
        "emart_auctions": emart_auctions_scout,
        "forum": forum_scout,
        "gdelt": gdelt_scout,
        "geopolitics": geopolitics_scout,
        "github_leaks": github_leaks_scout,
        "global_hpi": global_hpi_scout,
        "global_prime": global_prime_scout,
        "gnews": gnews_scout,
        "gtrends": gtrends_scout,
        "houston_permits": houston_permits_scout,
        "industry_research": industry_research_scout,
        "ir_pages": ir_pages_scout,
        "linkedin": linkedin_scout,
        "lme_metals": lme_metals_scout,
        "macro": macro_scout,
        "materials": materials_scout,
        "nasdaq_dubai": nasdaq_dubai_scout,
        "newsletters": newsletters_scout,
        "page_hash": page_hash_scout,
        "policy_wire": policy_wire_scout,
        "proptech_releasenotes": proptech_releasenotes_scout,
        "proptech": proptech_scout,
        "proptech_vc": proptech_vc_scout,
        "reddit": reddit_scout,
        "sanctions": sanctions_scout,
        "sec_edgar": sec_edgar_scout,
        "sukuk": sukuk_scout,
        "sydney_auctions": sydney_auctions_scout,
        "trade_press": trade_press_scout,
        "trakheesi": trakheesi_scout,
        "uae_cement": uae_cement_scout,
        "uae_press": uae_press_scout,
        "us_permits": us_permits_scout,
        "wayback_diff": wayback_diff_scout,
        "youtube": youtube_scout,
    }

    # Optional subset filter so the user can probe specific scouts fast.
    if only:
        want = {s.strip().lower() for s in only.split(",") if s.strip()}
        SCOUTS = {k: v for k, v in SCOUTS.items() if k.lower() in want}
        if not SCOUTS:
            return {"error": f"No scouts matched: {only}",
                    "available": list(SCOUTS.keys())}

    async def _probe_one(name: str, mod):
        started = _time.time()
        try:
            # linkedin_scout has a force kwarg to bypass its morning
            # gate; others have plain (limit=N) signature.
            if name == "linkedin":
                rows = await _asyncio.wait_for(
                    mod.run(limit=20, force=True), timeout=timeout_s
                )
            else:
                rows = await _asyncio.wait_for(
                    mod.run(limit=20), timeout=timeout_s
                )
        except _asyncio.TimeoutError:
            return name, {
                "raw_count": 0, "duration_s": round(_time.time() - started, 2),
                "regions": {}, "first_titles": [],
                "error": f"timeout after {timeout_s}s",
            }
        except Exception as e:
            return name, {
                "raw_count": 0, "duration_s": round(_time.time() - started, 2),
                "regions": {}, "first_titles": [],
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
        rows = rows or []
        regions: dict[str, int] = {}
        titles: list[str] = []
        for r in rows:
            rj = r.get("raw_json") if isinstance(r.get("raw_json"), dict) else {}
            reg = rj.get("region") or "(null)"
            regions[reg] = regions.get(reg, 0) + 1
            t = (r.get("title") or "").strip()
            if t and len(titles) < 3:
                titles.append(t[:140])
        return name, {
            "raw_count": len(rows),
            "duration_s": round(_time.time() - started, 2),
            "regions": regions,
            "first_titles": titles,
            "error": None,
        }

    overall_started = _time.time()
    results = await _asyncio.gather(
        *[_probe_one(n, m) for n, m in SCOUTS.items()],
        return_exceptions=False,
    )
    scouts = dict(results)

    # Roll up
    with_items = sum(1 for v in scouts.values() if v["raw_count"] > 0)
    zero = sum(1 for v in scouts.values() if v["raw_count"] == 0 and not v["error"])
    errored = sum(1 for v in scouts.values() if v["error"])
    by_region: dict[str, int] = {}
    for v in scouts.values():
        for r, c in v["regions"].items():
            by_region[r] = by_region.get(r, 0) + c

    from datetime import datetime as _dt, timezone as _tz
    return {
        "ran_at": _dt.now(_tz.utc).isoformat(),
        "duration_total_s": round(_time.time() - overall_started, 2),
        "summary": {
            "scouts_with_items": with_items,
            "scouts_zero":       zero,
            "scouts_errored":    errored,
            "by_region":         by_region,
        },
        "scouts": scouts,
    }


@app.post("/run/ingest")
async def run_ingest(force_linkedin: int = 0) -> dict:
    """Run just the ingest fan-out (no brief / no CEO scan).
    Returns the per-stage count dict from ingest_flow.run().

    Query params:
      ?force_linkedin=1  bypass the linkedin_scout's UTC morning-only
                          gate. Useful for ad-hoc debugging when the
                          Leader Feed tab is empty mid-day.
    """
    try:
        return await ingest_flow.run(force_linkedin=bool(force_linkedin))
    except Exception as e:
        log.exception("manual ingest failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ChangeTower webhook receiver removed 2026-04-26 — the free tier
# paywalls webhooks. Replaced by agents/scouts/page_hash_scout.py
# which performs the same competitor-page diff detection in-process,
# free, no third-party dependency.


# ─── Sobha portfolio (MIS) admin endpoints ─────────────────────────
# The Sobha MIS xlsx is the source of truth for every LLM stage's
# Sobha-context grounding. Sync via either:
#   (a) CLI:  python scripts/sync_sobha_mis.py --file <path>
#   (b) HTTP: POST /admin/sync-sobha-mis (multipart upload)

@app.get("/api/markets-brief")
async def markets_brief() -> dict:
    """Markets-page brief: live market snapshot + capital/macro synthesis
    + latest 3 capital-tagged signals. Frontend renders into the Markets
    page synth block.

    Falls through to whichever pieces are available — never blocks on
    one missing component.
    """
    out: dict = {"as_of": datetime.now(timezone.utc).isoformat()}

    # 1. Live market snapshot
    try:
        from tools import market_snapshot as _ms
        snap = await _ms.get_snapshot()
        out["market_line"] = snap.get("one_liner") or ""
        out["snapshot_rows"] = snap.get("rows") or []
    except Exception as e:
        log.info("markets-brief: snapshot failed: %s", e)
        out["market_line"] = ""
        out["snapshot_rows"] = []

    # 2. Latest brief's `markets` category synthesis (already produced by
    #    daily_brief_flow). May be "Synthesis unavailable." when the
    #    fast-model call failed — frontend handles either case.
    #    2026-04-29: also lift `bullets` so the Markets page can render
    #    the same rich themed-brief layout as Construction/Tech/Global
    #    (with Analyze ↗ button + inline source chips).
    try:
        latest = (
            sb.client().table("daily_briefs")
            .select("synthesis,day")
            .order("day", desc=True).limit(1).execute().data or [None]
        )[0] or {}
        synth = (latest.get("synthesis") or {}).get("markets") or {}
        out["synthesis"]       = (synth.get("synthesis") or "").strip()
        out["bullets"]         = synth.get("bullets") or []
        out["synthesis_count"] = synth.get("count") or 0
        out["synthesis_day"]   = latest.get("day")
    except Exception as e:
        log.info("markets-brief: synthesis fetch failed: %s", e)
        out["synthesis"] = ""
        out["bullets"]   = []

    # 3. True market signals only (2026-04-29 fix). The previous filter
    #    `decision_tag in ('capital','regulatory')` was too broad — it
    #    pulled RE acquisitions (Aldar buys KEZAD etc.) onto the Markets
    #    page even though those are competitor moves, not market data.
    #    Those RE deals already render on MD Scan + Developer Radar.
    #
    #    Strict filter:
    #      a) category='capital_markets' (eibor, sukuk, FX, rates, MAG-7
    #         spillover — set by category_hint at promotion)
    #      b) OR headline matches market-specific vocabulary (rate /
    #         yield / spread / sukuk / EIBOR / Treasury / Fed / DXY / etc.)
    #         — covers signals that landed before the keystone routing.
    try:
        from datetime import timedelta as _td
        seven_d = (datetime.now(timezone.utc) - _td(days=7)).isoformat()
        # Pull the over-set first; filter Python-side because PostgREST's
        # `or_` chaining gets unwieldy when you mix .in_() with .ilike().
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,decision_tag,dev_slug,region,last_seen_at,urls,corroboration_count")
            .gte("last_seen_at", seven_d)
            .eq("archived", False)
            .order("last_seen_at", desc=True).limit(40).execute().data or []
        )
        # Allowlist of headline tokens that mark a real market signal
        # even when the classifier didn't set category=capital_markets.
        MARKET_TOKENS = (
            "eibor", "sukuk", "treasury", "yield", "bond", "spread",
            "rate cut", "rate hike", "fed", "fomc", "dxy", "brent",
            "wti", "crude", "gold ", "bitcoin", " btc ", " eth ",
            "mortgage rate", "cds spread", "rating action", "downgrade",
            "upgrade", "fitch", "moody", "s&p ", "treasury yield",
            "ipo", "ipo'd", "share buyback", "dividend", "central bank",
            "fx", "usd/", "aed/", "inr/", "gbp/", "eur/", "cny/", "rub/",
        )
        kept = []
        for r in rows:
            cat = (r.get("category") or "").lower()
            head = ((r.get("headline") or "") + " " + (r.get("dek") or "")).lower()
            if cat in ("capital_markets", "capital_flow"):
                kept.append(r); continue
            if any(tok in head for tok in MARKET_TOKENS):
                kept.append(r); continue
        out["signals"] = kept[:5]
    except Exception as e:
        log.info("markets-brief: signals failed: %s", e)
        out["signals"] = []

    # 4. signal_url_index — id → url. Phase F2 inline source chips.
    try:
        ids = [int(s["id"]) for s in out.get("signals") or [] if s.get("id") is not None]
        out["signal_url_index"] = _build_signal_url_index(ids)
    except Exception as e:
        log.info("markets-brief: url-index failed: %s", e)
        out["signal_url_index"] = {}

    # 5. Markets-empty fallback: when the LLM returned the "Bucket thin
    #    today" placeholder, surface the live market_line as the bullet
    #    so the page reads as a snapshot, not a dead block. Stakeholder
    #    feedback 2026-04-30: "either show summary of what's changed in
    #    markets". The market_line is exactly that — S&P/Brent/DXY etc.
    bs = out.get("bullets") or []
    is_placeholder = (
        not bs or
        (len(bs) == 1 and (
            (isinstance(bs[0], dict) and (bs[0].get("text") or "").strip().lower().startswith("bucket thin"))
            or (isinstance(bs[0], str) and bs[0].strip().lower().startswith("bucket thin"))
        ))
    )
    if is_placeholder and out.get("market_line"):
        out["bullets"] = [{"text": out["market_line"], "signal_ids": []}]
        out["synthesis"] = out["market_line"]
        out["synthesis_count"] = 1

    return out


@app.get("/api/construction-snapshot")
async def construction_snapshot() -> dict:
    """Yahoo-backed price grid for the Construction Economics page.

    Pulls commodity equity proxies (SLX/CPER/VALE/AA/CX/EXP/WY/FCX/NUE/URI),
    UAE listed RE proxies, and container-shipping equities (ZIM/DAC/GSL/
    CMRE/STNG). Mirrors the Yahoo path market_snapshot already uses, so
    we reuse the cached snapshot's rows and just regroup for the tabs
    the construction page expects.

    No login required (Yahoo v8 is free) — addresses 2026-04-30 ask
    "is there any other way to show this data" beyond TradingView.
    """
    out: dict = {"as_of": datetime.now(timezone.utc).isoformat(),
                 "tabs": {"commodity": [], "uae_re": [], "shipping": []}}
    try:
        from tools import market_snapshot as _ms
        snap = await _ms.get_snapshot()
        rows = snap.get("rows") or []
    except Exception as e:
        log.info("construction-snapshot: fetch failed: %s", e)
        return out

    def _fmt(label: str, value: float) -> str:
        try:
            return _ms._format_value(label, value)  # type: ignore[attr-defined]
        except Exception:
            return f"{value:,.2f}"

    for r in rows:
        g = r.get("group")
        # Construction equity proxies + commodities
        if g in ("construction", "commodity"):
            out["tabs"]["commodity"].append({
                "label":      r.get("label"),
                "ticker":     r.get("ticker"),
                "value":      r.get("value"),
                "value_str":  _fmt(r.get("label", ""), r.get("value") or 0),
                "change_pct": r.get("change_pct"),
            })
        elif g == "uae_re":
            out["tabs"]["uae_re"].append({
                "label":      r.get("label"),
                "ticker":     r.get("ticker"),
                "value":      r.get("value"),
                "value_str":  _fmt(r.get("label", ""), r.get("value") or 0),
                "change_pct": r.get("change_pct"),
            })
        elif g == "shipping":
            out["tabs"]["shipping"].append({
                "label":      r.get("label"),
                "ticker":     r.get("ticker"),
                "value":      r.get("value"),
                "value_str":  _fmt(r.get("label", ""), r.get("value") or 0),
                "change_pct": r.get("change_pct"),
            })
    return out


@app.get("/api/construction-innovation")
async def construction_innovation() -> dict:
    """ConTech innovation rail — prefab / modular / 3D-print / robotics /
    fast-build records / AI-design / BIM / digital twin.

    The Construction Econ themed brief pulls categories=['materials',
    'supply'] which surfaces commodity prices but misses the innovation
    half ("Abu Dhabi built a tower in 14 days"). This rail catches
    those by token-matching across ALL categories, since ConTech
    innovation news typically gets classified as 'tech' or 'other'.

    Returns last 21 days of signals whose headline+dek matches an
    innovation token, sorted high→medium→low.
    """
    out: dict = {"as_of": datetime.now(timezone.utc).isoformat(), "rows": []}
    INNOVATION_TOKENS = (
        # Methodology
        "prefab", "prefabricat", "modular", "modular construct",
        "off-site", "offsite construct", "panelized",
        "3d print", "3d-print", "3d printed", "additive manufact",
        "concrete print", "robotic construct", "construction robot",
        # Speed / records
        "built in", "completed in", "days to build", "fast build",
        "fastest construct", "speed build", "expedited construct",
        "rapid build", "tower in",
        # Software / digital
        "bim ", "digital twin", "ai design", "ai-design", "generative design",
        "computational design", "matterport", "autodesk forge",
        "procore", "revit", "rhino grasshopper",
        # Smart-construction equipment
        "exoskeleton", "construction drone", "self-driving crane",
        "autonomous excavator", "robotic mason", "bricklaying robot",
        # Sustainability / materials innovation
        "low-carbon concrete", "geopolymer", "carbon capture concrete",
        "mass timber", "cross-laminated timber", "clt",
        "self-healing concrete", "recycled aggregate",
    )
    try:
        from datetime import timedelta as _td
        since = (datetime.now(timezone.utc) - _td(days=21)).isoformat()
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,decision_tag,region,country_code,last_seen_at,urls,corroboration_count")
            .eq("archived", False)
            .gte("last_seen_at", since)
            .order("last_seen_at", desc=True).limit(60).execute().data or []
        )
    except Exception as e:
        log.info("construction-innovation: fetch failed: %s", e)
        return out
    matched = []
    for r in rows:
        haystack = ((r.get("headline") or "") + " " + (r.get("dek") or "")).lower()
        if any(tok in haystack for tok in INNOVATION_TOKENS):
            matched.append(r)
    pri_rank = {"high": 0, "medium": 1, "med": 1, "low": 2}
    matched.sort(key=lambda s: pri_rank.get((s.get("priority") or "").lower(), 9))
    out["rows"] = matched[:8]
    return out


@app.get("/api/global-hpi")
async def global_hpi() -> dict:
    """Latest HPI level + MoM% per tracked city for the Global RE Pulse page.

    Reads metrics_daily where metric_key starts with 'hpi_' (populated by
    global_hpi_scout). Returns a sorted list of {city, level, pct_change,
    captured_at, source_url} ready for table rendering.

    2026-04-30 (audit-round 4): page was sparse — just the synth bullets
    and nothing else. This adds a real data block driven by data we
    already collect.
    """
    out: dict = {"as_of": datetime.now(timezone.utc).isoformat(), "rows": []}
    try:
        from datetime import timedelta as _td
        since = (datetime.now(timezone.utc) - _td(days=90)).isoformat()
        rows = (
            sb.client().table("metrics_daily")
            .select("metric_key,value_num,captured_at,raw_json,source")
            .like("metric_key", "hpi_%")
            .gte("captured_at", since)
            .order("captured_at", desc=True).limit(50).execute().data or []
        )
    except Exception as e:
        log.info("global-hpi: fetch failed: %s", e)
        return out
    # Take latest row per city (metric_key is already unique per city).
    seen: set[str] = set()
    out_rows: list[dict] = []
    # Display ordering — Sobha-relevant first.
    DISPLAY_ORDER = ["hpi_uk_london", "hpi_singapore", "hpi_nyc_manhattan",
                     "hpi_hong_kong", "hpi_india_metro"]
    LABEL_MAP = {
        "hpi_uk_london":     "London / UK",
        "hpi_singapore":     "Singapore (URA)",
        "hpi_nyc_manhattan": "NYC Manhattan luxury",
        "hpi_hong_kong":     "Hong Kong",
        "hpi_india_metro":   "India metros",
    }
    by_key: dict[str, dict] = {}
    for r in rows:
        key = r.get("metric_key")
        if not key or key in by_key:
            continue
        by_key[key] = r
    for key in DISPLAY_ORDER + [k for k in by_key if k not in DISPLAY_ORDER]:
        r = by_key.get(key)
        if not r or key in seen:
            continue
        seen.add(key)
        rj = r.get("raw_json") or {}
        out_rows.append({
            "key":          key,
            "label":        LABEL_MAP.get(key, rj.get("label") or key.replace("hpi_", "")),
            "level":        r.get("value_num"),
            "pct_change":   rj.get("pct_change"),
            "captured_at":  r.get("captured_at"),
            "country":      rj.get("country"),
            "source_url":   rj.get("url"),
            "source_title": rj.get("title"),
        })
    out["rows"] = out_rows
    return out


@app.get("/api/global-geopolitics")
async def global_geopolitics() -> dict:
    """Recent geopolitics signals filtered to RE-impact for Global RE Pulse.

    Pulls last 14 days of signals tagged category='geopolitics' (set by
    geopolitics_scout via category_hint at promotion). Sorted high →
    medium → low. Caps at 8 rows for the rail.
    """
    out: dict = {"as_of": datetime.now(timezone.utc).isoformat(), "rows": []}
    try:
        from datetime import timedelta as _td
        since = (datetime.now(timezone.utc) - _td(days=14)).isoformat()
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,decision_tag,region,country_code,last_seen_at,urls,corroboration_count")
            .eq("category", "geopolitics")
            .eq("archived", False)
            .gte("last_seen_at", since)
            .order("last_seen_at", desc=True).limit(20).execute().data or []
        )
    except Exception as e:
        log.info("global-geopolitics: fetch failed: %s", e)
        return out
    pri_rank = {"high": 0, "medium": 1, "med": 1, "low": 2}
    rows.sort(key=lambda s: pri_rank.get((s.get("priority") or "").lower(), 9))
    out["rows"] = rows[:8]
    return out


def _build_signal_url_index(signal_ids: list[int]) -> dict[str, str]:
    """Return {id: first_url} for the named signals.

    Powers the inline ↗ source chips on the dashboard (Phase F2).
    Bullets carry signal_ids; the dashboard resolves each to a URL via
    this index. One round-trip to Supabase per brief render — cheap.

    Returns string-keyed dict because JSON object keys are always strings,
    and the frontend reads url_index[String(signal_id)].
    """
    if not signal_ids:
        return {}
    try:
        rows = (
            sb.client().table("signals")
            .select("id,urls")
            .in_("id", signal_ids)
            .execute().data or []
        )
    except Exception as e:
        log.info("signal_url_index: fetch failed: %s", e)
        return {}
    out: dict[str, str] = {}
    for r in rows:
        sid = r.get("id")
        urls = r.get("urls") or []
        if sid is not None and urls:
            out[str(sid)] = urls[0]
    return out


_WEB_FILL_QUERIES: dict[str, dict[str, Any]] = {
    "construction": {
        # 2026-04-30 (round 5): MD wants ConTech INNOVATION, not just
        # commodity prices. Prefab / modular / 3D-print / robotics /
        # speed-build records / BIM / digital twin breakthroughs.
        # The materials side stays in the keyword set so cost-side
        # signals still surface, but the innovation half is now the
        # priority.
        "query":   "construction technology innovation prefabricated modular 3D printed building robotics fast build days speed record BIM digital twin AI design Dubai UAE materials cost steel cement 2026 this week",
        "domains": ["constructiondive.com", "bdcnetwork.com",
                    "enr.com", "theconstructionindex.co.uk",
                    "aecmag.com", "archpaper.com",
                    "archdaily.com", "dezeen.com",
                    "constructionweekonline.com", "meed.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "spglobal.com", "argusmedia.com", "fastmarkets.com"],
        "label":   "Construction Economics & ConTech",
    },
    "tech": {
        # 2026-04-30 (round 6): PropTech + Frontier AI only. ConTech
        # moved to its own dedicated query for Construction Economics
        # page (see "construction" entry). This query is for: real-
        # estate sales/discovery/financing/management software AND
        # frontier AI moves the world is talking about.
        "query":   "PropTech real estate technology Zillow Compass Opendoor CoStar Bayut Anthropic Claude OpenAI GPT model launch funding round Nvidia DeepMind AI breakthrough 2026 this week",
        "domains": [
            # Frontier AI press
            "techcrunch.com", "theinformation.com", "venturebeat.com",
            "wired.com", "theverge.com", "arstechnica.com",
            "anthropic.com", "openai.com", "deepmind.com",
            "deepmind.google", "blog.google", "nvidia.com",
            "huggingface.co", "perplexity.ai", "mistral.ai",
            # PropTech press
            "propmodo.com", "therealdeal.com", "bisnow.com",
            "globest.com", "inman.com", "crunchbase.com",
            # MENA tech
            "wamda.com", "menabytes.com",
            # General business
            "ft.com", "bloomberg.com", "reuters.com", "wsj.com",
        ],
        "label":   "Tech & AI",
    },
    "global": {
        "query":   "London Singapore NYC Hong Kong prime residential property HNWI migration 2026 this week",
        "domains": ["knightfrank.com", "savills.com", "henleyglobal.com",
                    "ft.com", "bloomberg.com", "wsj.com",
                    "mansionglobal.com", "thenationalnews.com"],
        "label":   "Global RE pulse",
    },
    "markets": {
        "query":   "EIBOR sukuk yield UAE bond MAG-7 Dubai capital markets this week 2026",
        "domains": ["nasdaqdubai.com", "mubasher.info", "zawya.com",
                    "bloomberg.com", "reuters.com", "ft.com",
                    "thenationalnews.com", "khaleejtimes.com",
                    "spglobal.com", "tradingeconomics.com"],
        "label":   "Markets & capital",
    },
}


# Cache web force-fill results so we don't burn 8-16 Tavily credits
# per dashboard page-view. TTL = 6h; the curated weekly roundup
# doesn't move that fast. Process-local — Railway runs single-instance
# so this is enough; if scaled out we'd need Redis. (Audit-round 4
# fix: 900 credits per pipeline run was untenable.)
_WEB_FILL_CACHE: dict[str, dict] = {}
_WEB_FILL_TTL_SEC = 6 * 60 * 60  # 6 hours


async def _web_force_fill(slug: str) -> dict:
    """Generate a 'this week in [bucket]' summary via Tavily/Brave when
    the themed-page bucket has no native signals (2026-04-30).

    Saves the MD from staring at "Bucket thin today" — instead they see
    a curated web roundup, marked clearly as web-research vs Sobha-tagged.
    Returns {bullets: [{text, signal_ids:[]}], synthesis: str,
             web_research: True}.

    Cached for 6h per slug to keep Tavily credit usage sane.
    """
    cfg = _WEB_FILL_QUERIES.get(slug)
    if not cfg:
        return {}
    # Cache hit — dashboard re-renders should NOT burn Tavily credits.
    import time as _time
    cached = _WEB_FILL_CACHE.get(slug)
    if cached and (_time.time() - cached.get("_cached_at", 0)) < _WEB_FILL_TTL_SEC:
        return cached.get("payload") or {}
    from tools import tavily_search
    try:
        # 2026-04-30 audit-round 4: dropped from search_depth=advanced
        # (2 credits/call) to "basic" (1 credit/call). Halves the bill;
        # for the curated weekly roundup the extra depth wasn't paying
        # for itself. Also reduced max_results 6 → 4.
        results = await tavily_search.search(
            cfg["query"], max_results=4,
            topic="news", search_depth="basic",
            include_domains=cfg["domains"], days=10,
        )
    except Exception as e:
        log.info("[web-fill:%s] tavily failed: %s", slug, e)
        return {}
    if not results:
        return {}
    # Synthesise into bullets via the existing fast LLM path. Each bullet
    # is a single fact from one search hit; the LLM is instructed to be
    # neutral (no Sobha-anchor analysis — that lives behind Analyze).
    from tools.nvidia_llm import chat_json
    # Number each web hit so the LLM can cite by index. We map each
    # numbered cite back to a synthetic negative signal_id and a URL,
    # which the existing signal_url_index machinery already handles
    # → bullets get their inline ↗ source chip for free.
    numbered = [
        {
            "idx":     i + 1,
            "title":   (r.get("title") or "")[:200],
            "snippet": (r.get("content") or "")[:280],
            "url":     r.get("url"),
        }
        for i, r in enumerate(results[:6])
    ]
    SYS = (
        "You write factual one-line bullets summarising real-estate news. "
        "Each bullet ≤ 20 words. No interpretation. Lead with entity + "
        "number where present (e.g. 'Knight Frank: London PCL up 1.2% MoM').\n"
        "Each input hit is numbered 1..N. Output JSON shape:\n"
        '  {"bullets": [{"text": "...", "src": [1,2]}, ...]}\n'
        "where src lists the input-hit indices the bullet draws from. "
        "Use 1-2 indices per bullet. Skip duplicates."
    )
    try:
        parsed = await chat_json(
            [
                {"role": "system", "content": SYS},
                {"role": "user",   "content": (
                    f"Topic: {cfg['label']}. Recent web hits:\n{numbered}\n"
                    "Write 4-6 factual bullets, each with src indices."
                )},
            ],
            max_tokens=500,
            role="fast",
        )
        bullets_raw = parsed.get("bullets") or []
    except Exception as e:
        log.info("[web-fill:%s] LLM failed: %s", slug, e)
        bullets_raw = []

    # Synthetic signal_ids: use negative numbers per slug so they never
    # collide with real Supabase signal IDs. Hash-based to keep stable
    # across re-renders within a request.
    base = -1_000_000 - (abs(hash(slug)) % 9000) * 100
    signal_url_index: dict[str, str] = {}
    bullets: list[dict[str, Any]] = []
    for raw in bullets_raw[:6]:
        if isinstance(raw, dict):
            text = str(raw.get("text") or "").strip()
            src_idxs = [int(i) for i in (raw.get("src") or []) if isinstance(i, (int, str)) and str(i).isdigit()]
        else:
            text, src_idxs = str(raw).strip(), []
        if not text:
            continue
        signal_ids: list[int] = []
        for s in src_idxs[:3]:
            if 1 <= s <= len(numbered):
                hit = numbered[s - 1]
                if not hit.get("url"):
                    continue
                synth_id = base - (len(bullets) * 10 + s)
                signal_url_index[str(synth_id)] = hit["url"]
                signal_ids.append(synth_id)
        # Fallback — if the LLM didn't attribute, pick the closest hit
        # by title overlap so every bullet still gets at least one ↗.
        if not signal_ids and numbered:
            best, best_score = numbered[0], 0
            tw = set(text.lower().split())
            for hit in numbered:
                hw = set((hit.get("title") or "").lower().split())
                score = len(tw & hw)
                if score > best_score:
                    best, best_score = hit, score
            if best.get("url"):
                synth_id = base - (len(bullets) * 10)
                signal_url_index[str(synth_id)] = best["url"]
                signal_ids.append(synth_id)
        bullets.append({"text": text, "signal_ids": signal_ids})

    if not bullets:
        # Last-resort fallback: title strings with their own URL
        # attached so the MD still gets clickable sources.
        for i, r in enumerate(results[:6]):
            title = (r.get("title") or "")[:200]
            url = r.get("url")
            if not title or not url:
                continue
            synth_id = base - (i * 10)
            signal_url_index[str(synth_id)] = url
            bullets.append({"text": title, "signal_ids": [synth_id]})

    payload = {
        "bullets":         bullets,
        "synthesis":       " · ".join(b["text"] for b in bullets),
        "synthesis_count": len(bullets),
        "web_research":    True,
        "web_signal_url_index": signal_url_index,
        "web_results":     [
            {"title": r.get("title"), "url": r.get("url"),
             "content": (r.get("content") or "")[:300]}
            for r in results[:6]
        ],
        "web_query":       cfg["label"],
    }
    # Stash in cache so subsequent page-views don't re-burn credits.
    _WEB_FILL_CACHE[slug] = {"_cached_at": _time.time(), "payload": payload}
    return payload


async def _themed_page_brief(
    bucket_key: str,
    categories: list[str],
    decision_tags: list[str] | None = None,
    days_back: int = 14,
    signal_limit: int = 12,
    subcategories: list[str] | None = None,
    legacy_keyword_allow: tuple[str, ...] | None = None,
    legacy_keyword_deny: tuple[str, ...] | None = None,
) -> dict:
    """Shared loader for the Construction / Tech / Global themed pages.

    Returns:
      - bullets:        from daily_briefs.synthesis[bucket_key].bullets (3-6 items)
      - synthesis:      legacy paragraph (back-compat)
      - synthesis_count, synthesis_day: provenance
      - signals:        latest N signals matching `categories` (or decision_tags
                        when supplied — used for legacy buckets that pre-date
                        the new category enum)

    2026-04-30 round 7: signal routing now uses (category, subcategory)
    set by the LLM classifier. Pass `subcategories` to filter further;
    rows whose subcategory is NULL (legacy, pre-migration-019) match
    the optional keyword fallback (`legacy_keyword_allow`/`deny`) so we
    don't lose existing signals during the migration period.

    Mirrors the /api/markets-brief shape so the dashboard render path is
    parallel across all themed pages — no silos.
    """
    out: dict = {"as_of": datetime.now(timezone.utc).isoformat()}

    # 1. Latest brief's per-bucket synthesis (bullets + legacy paragraph).
    try:
        latest = (
            sb.client().table("daily_briefs")
            .select("synthesis,day")
            .order("day", desc=True).limit(1).execute().data or [None]
        )[0] or {}
        synth = (latest.get("synthesis") or {}).get(bucket_key) or {}
        out["bullets"] = synth.get("bullets") or []
        out["synthesis"] = (synth.get("synthesis") or "").strip()
        out["synthesis_count"] = synth.get("count") or 0
        out["synthesis_day"] = latest.get("day")
    except Exception as e:
        log.info("%s-brief: synthesis fetch failed: %s", bucket_key, e)
        out["bullets"] = []
        out["synthesis"] = ""
        out["synthesis_count"] = 0

    # 2. Latest signals matching the page's categories (or decision_tags).
    try:
        from datetime import timedelta as _td
        since = (datetime.now(timezone.utc) - _td(days=days_back)).isoformat()
        # Try to select subcategory; fall back if migration 019 hasn't run.
        select_cols = ("id,headline,dek,priority,category,subcategory,"
                       "decision_tag,dev_slug,region,country_code,"
                       "last_seen_at,urls,corroboration_count")
        try:
            q = (
                sb.client().table("signals")
                .select(select_cols)
                .gte("last_seen_at", since)
                .eq("archived", False)
                .order("last_seen_at", desc=True)
                .limit(signal_limit * 4)  # over-fetch — bigger margin for sub-filter
            )
            if decision_tags:
                rows = q.in_("decision_tag", decision_tags).execute().data or []
            else:
                rows = q.execute().data or []
        except Exception as e:
            if "subcategory" in str(e).lower():
                # Pre-migration-019 row — re-query without the column.
                q = (
                    sb.client().table("signals")
                    .select(select_cols.replace(",subcategory", ""))
                    .gte("last_seen_at", since)
                    .eq("archived", False)
                    .order("last_seen_at", desc=True)
                    .limit(signal_limit * 4)
                )
                if decision_tags:
                    rows = q.in_("decision_tag", decision_tags).execute().data or []
                else:
                    rows = q.execute().data or []
                for r in rows:
                    r.setdefault("subcategory", None)
            else:
                raise
        # Category filter — primary gate.
        if categories:
            cats_set = set(categories)
            rows = [r for r in rows if (r.get("category") or "") in cats_set]
        # Subcategory routing (LLM-driven, round 7) — preferred path.
        # Rows with a subcategory get strict gating; rows where
        # subcategory is NULL fall through to the legacy keyword
        # filter so pre-migration signals still surface during the
        # migration window.
        if subcategories or legacy_keyword_allow or legacy_keyword_deny:
            sub_set = set(subcategories or [])
            allow = tuple(t.lower() for t in (legacy_keyword_allow or ()))
            deny = tuple(t.lower() for t in (legacy_keyword_deny or ()))
            kept: list[dict] = []
            for r in rows:
                sub = (r.get("subcategory") or "").strip().lower()
                head = ((r.get("headline") or "") + " "
                        + (r.get("dek") or "")).lower()
                if sub:
                    # LLM has decided — trust it.
                    if not sub_set or sub in sub_set:
                        kept.append(r)
                    continue
                # Legacy row (subcategory NULL) — keyword fallback.
                if deny and any(t in head for t in deny):
                    if not (allow and any(t in head for t in allow)):
                        continue
                if allow and not any(t in head for t in allow):
                    # Can be too strict — only deny if we have explicit allow.
                    # Without allow tokens, fall through (keep the row).
                    continue
                kept.append(r)
            rows = kept
        out["signals"] = rows[:signal_limit]
    except Exception as e:
        log.info("%s-brief: signals failed: %s", bucket_key, e)
        out["signals"] = []

    # 3. signal_url_index — id → url for inline ↗ source chips on bullets.
    #    Covers BOTH bullet-attributed signal_ids (from the brief synthesis)
    #    AND the signal cards rendered below. One Supabase round-trip.
    try:
        ids: set[int] = set()
        for b in out.get("bullets") or []:
            if isinstance(b, dict):
                for sid in b.get("signal_ids") or []:
                    try:
                        ids.add(int(sid))
                    except Exception:
                        pass
        for s in out.get("signals") or []:
            if s.get("id") is not None:
                try:
                    ids.add(int(s["id"]))
                except Exception:
                    pass
        out["signal_url_index"] = _build_signal_url_index(list(ids))
    except Exception as e:
        log.info("%s-brief: url-index failed: %s", bucket_key, e)
        out["signal_url_index"] = {}

    # 4. Force-fill via web research when the bucket is genuinely empty
    #    (no native signals AND no bucket-synth bullets). Saves the MD
    #    from staring at "Bucket thin today" — gives them a curated
    #    weekly roundup pulled via Tavily/Brave from trusted domains.
    no_signals = not (out.get("signals") or [])
    bullets_now = out.get("bullets") or []
    # Placeholder = LLM returned "Bucket thin today." (or similar empty
    # state) — treat as effectively empty so force-fill can trigger.
    def _is_placeholder(bs: list) -> bool:
        if not bs: return True
        if len(bs) > 1: return False
        b0 = bs[0]
        text = (b0.get("text") if isinstance(b0, dict) else str(b0)) or ""
        t = text.strip().lower()
        return t in ("", "bucket thin today.", "bucket thin today",
                     "no recent signals.", "no recent signals")
    no_bullets = _is_placeholder(bullets_now)
    if no_signals and no_bullets:
        web = await _web_force_fill(bucket_key)
        if web:
            out["bullets"]         = web.get("bullets") or []
            out["synthesis"]       = web.get("synthesis") or ""
            out["synthesis_count"] = web.get("synthesis_count") or 0
            out["web_research"]    = True
            out["web_results"]     = web.get("web_results") or []
            out["web_query"]       = web.get("web_query") or ""
            # Merge the web-bullet synthetic ID → URL map into the
            # primary signal_url_index so the existing frontend cite
            # resolver finds them transparently. Keys are stringified
            # negative ints; collide-free with real signal IDs.
            merged = dict(out.get("signal_url_index") or {})
            merged.update(web.get("web_signal_url_index") or {})
            out["signal_url_index"] = merged

    return out


@app.get("/api/construction-brief")
async def construction_brief() -> dict:
    """Construction Economics page — material prices + ConTech innovation.

    2026-04-30 round 7: LLM-driven subcategory routing replaces keyword
    filters. The classifier picks materials_cost / materials_innovation
    for the materials category, and ConTech innovation lives on the
    dedicated /api/construction-innovation rail. Legacy rows
    (subcategory=NULL) fall through to a keyword guard that drops the
    old land-deal noise.

    Always-merge web force-fill so Construction Dive / ENR / BD+C
    headlines surface every visit.
    """
    out = await _themed_page_brief(
        bucket_key="construction",
        categories=["materials", "supply"],
        signal_limit=15,
        subcategories=["materials_cost", "materials_innovation"],
        legacy_keyword_deny=("land auction", "land plot", "mansion sells",
                              "branded residences", "tax exemption"),
    )
    # Always-merge force-fill (same pattern as tech_brief).
    try:
        web = await _web_force_fill("construction")
        if web and web.get("bullets"):
            existing = out.get("bullets") or []
            def _is_placeholder_bullets(bs):
                if not bs: return True
                if len(bs) == 1:
                    b0 = bs[0]
                    text = (b0.get("text") if isinstance(b0, dict) else str(b0)) or ""
                    return text.strip().lower().startswith("bucket thin")
                return False
            wb = web.get("bullets") or []
            if _is_placeholder_bullets(existing):
                out["bullets"] = wb[:8]
            else:
                seen, merged = set(), []
                for b in existing[:4]:
                    text = (b.get("text") if isinstance(b, dict) else str(b)) or ""
                    k = text.lower()[:60]
                    if k and k not in seen:
                        seen.add(k); merged.append(b)
                for b in wb[:6]:
                    text = (b.get("text") if isinstance(b, dict) else str(b)) or ""
                    k = text.lower()[:60]
                    if k and k not in seen:
                        seen.add(k); merged.append(b)
                out["bullets"] = merged
            out["web_research"] = True
            out["web_results"]  = (out.get("web_results") or []) + (web.get("web_results") or [])
            out["web_query"]    = web.get("web_query") or out.get("web_query") or ""
            mi = dict(out.get("signal_url_index") or {})
            mi.update(web.get("web_signal_url_index") or {})
            out["signal_url_index"] = mi
            out["synthesis_count"] = len(out["bullets"])
    except Exception as e:
        log.info("construction-brief: force-fill mix failed: %s", e)
    return out


@app.get("/api/tech-brief")
async def tech_brief() -> dict:
    """Tech & AI page — PropTech / ConTech / AI design / tokenized RE.

    Category: tech (set by proptech_scout + classifier when news mentions
    PropTech / ConTech / AI / digital twin / tokenized RE).

    2026-04-30 stakeholder feedback: the page was filling up with
    RE-stake news (Leela 25% in Palm Jumeirah, Prestige+Autodesk
    partnership, etc.) because the classifier flagged them tech for
    mentioning 'digital'/'tokenization'/'AI'. The MD wants real tech
    news — TechCrunch funding rounds, Anthropic/OpenAI moves, AI
    research, ConTech innovation. Two changes:
      1. Drop signals where dev_slug is set — those are competitor
         moves, not tech, and they belong on MD Scan.
      2. Always-merge web force-fill (TechCrunch + Crunchbase via
         Tavily) so AI/PropTech-VC stories show up on every visit,
         not only when the bucket is empty.
    """
    # 2026-04-30 round 7: LLM subcategory routing replaces all the
    # keyword filters. Tech page = proptech + frontier_ai only. ConTech
    # lives on the Construction page's innovation rail. Legacy rows
    # (subcategory NULL) fall through to a tightened keyword fallback.
    out = await _themed_page_brief(
        bucket_key="tech",
        categories=["tech"],
        signal_limit=20,
        subcategories=["proptech", "frontier_ai"],
        legacy_keyword_allow=(
            # Frontier AI players
            "anthropic", "openai", "claude", "gpt", "deepmind", "gemini",
            "nvidia", "mistral", "cohere", "perplexity", "huggingface",
            "hugging face", "llama",
            # PropTech players
            "zillow", "redfin", "compass", "opendoor", "costar",
            "bayut", "property finder", "yardi", "realpage", "propmodo",
            "proptech", "tokenize", "tokenized", "smart home",
        ),
        legacy_keyword_deny=(
            "stake in", "branded residences", "land auction",
            "tax exemption", "real estate fund",
            # ConTech tokens — belong on Construction page only.
            "prefab", "modular construct", "3d print", "bim ",
            "digital twin", "robotic mason", "mass timber",
        ),
    )
    # Drop dev_slug-anchored signals (those are competitor news, MD
    # Scan territory). Applies regardless of subcategory.
    sigs = out.get("signals") or []
    out["signals"] = [s for s in sigs if not (s.get("dev_slug") or "").strip()]

    # Filter 2 — always-mix force-fill on the Tech page even when
    # native signals exist. The MD asked specifically for "more AI &
    # tech happening" — TechCrunch + Anthropic + OpenAI roundup
    # always surfaces. Existing native bullets stay first; web bullets
    # are appended underneath, capped to keep the synth cell readable.
    try:
        web = await _web_force_fill("tech")
        if web and web.get("bullets"):
            existing = out.get("bullets") or []
            # If existing is the placeholder, replace; else append.
            def _is_placeholder_bullets(bs):
                if not bs: return True
                if len(bs) == 1:
                    b0 = bs[0]
                    text = (b0.get("text") if isinstance(b0, dict) else str(b0)) or ""
                    return text.strip().lower().startswith("bucket thin")
                return False
            web_bullets = web.get("bullets") or []
            if _is_placeholder_bullets(existing):
                out["bullets"] = web_bullets[:8]
            else:
                # Mix: dedup by lowercase prefix; keep up to 4 native + 6 web.
                seen = set()
                merged: list = []
                for b in existing[:4]:
                    text = (b.get("text") if isinstance(b, dict) else str(b)) or ""
                    k = text.lower()[:60]
                    if k and k not in seen:
                        seen.add(k); merged.append(b)
                for b in web_bullets[:6]:
                    text = (b.get("text") if isinstance(b, dict) else str(b)) or ""
                    k = text.lower()[:60]
                    if k and k not in seen:
                        seen.add(k); merged.append(b)
                out["bullets"] = merged
            out["web_research"] = True
            out["web_results"]  = (out.get("web_results") or []) + (web.get("web_results") or [])
            out["web_query"]    = web.get("web_query") or out.get("web_query") or ""
            merged_idx = dict(out.get("signal_url_index") or {})
            merged_idx.update(web.get("web_signal_url_index") or {})
            out["signal_url_index"] = merged_idx
            out["synthesis_count"] = len(out["bullets"])
    except Exception as e:
        log.info("tech-brief: force-fill mix failed: %s", e)

    return out


@app.get("/api/global-brief")
async def global_brief() -> dict:
    """Global RE Pulse page — prime markets + geopolitics + capital flow.

    Categories: global_prime (London/Singapore/NYC/Miami/Monaco prime),
    geopolitics (RE-impact filtered), capital_flow (population/migration/
    LRS/ODI/FDI). Always-merges web force-fill (Knight Frank / Savills /
    Henley / FT / Mansion Global) so London/Singapore/NYC prime moves
    surface even when our scouts didn't catch a fresh one this cycle.
    """
    # 2026-04-30 round 7: subcategory routing for Global RE Pulse.
    # hpi_prime / hnwi_migration / capital_flow_other / geopolitics_re_impact.
    out = await _themed_page_brief(
        bucket_key="global",
        categories=["global_prime", "geopolitics", "capital_flow"],
        signal_limit=20,
        subcategories=["hpi_prime", "hnwi_migration", "capital_flow_other",
                       "geopolitics_re_impact"],
    )
    try:
        web = await _web_force_fill("global")
        if web and web.get("bullets"):
            existing = out.get("bullets") or []
            def _is_placeholder_bullets(bs):
                if not bs: return True
                if len(bs) == 1:
                    b0 = bs[0]
                    text = (b0.get("text") if isinstance(b0, dict) else str(b0)) or ""
                    return text.strip().lower().startswith("bucket thin")
                return False
            wb = web.get("bullets") or []
            if _is_placeholder_bullets(existing):
                out["bullets"] = wb[:8]
            else:
                seen, merged = set(), []
                for b in existing[:4]:
                    text = (b.get("text") if isinstance(b, dict) else str(b)) or ""
                    k = text.lower()[:60]
                    if k and k not in seen:
                        seen.add(k); merged.append(b)
                for b in wb[:6]:
                    text = (b.get("text") if isinstance(b, dict) else str(b)) or ""
                    k = text.lower()[:60]
                    if k and k not in seen:
                        seen.add(k); merged.append(b)
                out["bullets"] = merged
            out["web_research"] = True
            out["web_results"]  = (out.get("web_results") or []) + (web.get("web_results") or [])
            out["web_query"]    = web.get("web_query") or out.get("web_query") or ""
            mi = dict(out.get("signal_url_index") or {})
            mi.update(web.get("web_signal_url_index") or {})
            out["signal_url_index"] = mi
            out["synthesis_count"] = len(out["bullets"])
    except Exception as e:
        log.info("global-brief: force-fill mix failed: %s", e)

    # 2026-04-30 (audit-round 3): HPI MoM sanity guard — strip signals
    # whose headline / dek claims an MoM HPI move >5%. The scout-level
    # guard in global_hpi_scout._parse_move only catches HPI rows that
    # came through that scout; classifier-derived headlines from gnews/
    # gdelt (e.g. WSJ Manhattan luxury HPI) bypass it entirely. Implausible
    # MoM moves are almost always YoY mislabels — better to drop than to
    # show "NYC HPI up 14.4% MoM" on the front page.
    import re as _re
    _hpi_rx = _re.compile(r"hpi\b.*?(\d+(?:\.\d+)?)\s*%\s*(?:mom|m/m|month[\s\-]on[\s\-]month)", _re.I)
    _annual_tokens = ("yoy", "y/y", "year-on-year", "year on year",
                      "annual", "annualised", "annualized")
    def _is_implausible_hpi(text: str) -> bool:
        if not text: return False
        m = _hpi_rx.search(text)
        if not m: return False
        try:
            v = float(m.group(1))
        except Exception:
            return False
        if v <= 5.0: return False
        return not any(t in text.lower() for t in _annual_tokens)
    sigs = out.get("signals") or []
    out["signals"] = [
        s for s in sigs
        if not _is_implausible_hpi(((s.get("headline") or "") + " " + (s.get("dek") or "")))
    ]
    bullets_in = out.get("bullets") or []
    out["bullets"] = [
        b for b in bullets_in
        if not _is_implausible_hpi(b.get("text") if isinstance(b, dict) else str(b))
    ]
    return out


ANALYZE_SYSTEM = """You are the strategic-read layer for the MD of Sobha Realty.

The morning brief gave the MD a list of FACTUAL pointers (who did what, with what number). You are now invoked on demand when the MD clicks "Analyze ↗" on one of those pointers (or a bucket of them). Your job is to produce the so-what — the strategic read the brief deliberately withheld.

ARCHITECTURE — Sobha is the ANCHOR, not the SUBJECT:
- The signals describe EXTERNAL moves (competitor / regulator / capital / macro / supply / tech).
- Your read describes how those moves reposition Sobha's KPI / launch / pricing / capital decisions.
- You may ALSO flag what's missing — "Need: <fact>. Delegate to: <team>." — when an interpretation depends on data the signals don't carry.

Output: a JSON object.
{
  "bullets": [
    {"text": "≤ 18 word strategic read", "signal_ids": [<int>, ...]},
    ...
  ],
  "data_gaps": ["≤ 14 word gap statement", ...],
  "counterfactuals": ["≤ 18 word disconfirming-evidence prompt", ...],
  "stance": "opportunity|risk|watch|mixed"
}

Rules:
  - 2-5 bullets. Lead with the most decision-relevant read.
  - Each bullet must reference at least one signal_id from the input.
  - Quantify when the signals support it (PSF, %, units, AED).
  - Sobha is the ANCHOR — name a Sobha asset / community / KPI when relevant.
    Use only entities present in the input or in the portfolio_summary.
  - 0-3 data_gaps. Use only when a real gap blocks the read.
  - 1-3 counterfactuals — what would have to be true for this read to be wrong?
    Surface the disconfirming evidence the system should be hunting for.
    Example: "Sobha Hartland II resale velocity unchanged WoW" or
    "DLD foreign-buyer share didn't move in past 30 days".
  - stance is one word; "mixed" is allowed when bullets carry both opportunity and risk.
  - JSON ONLY, no fences, no preamble."""


@app.post("/api/analyze")
async def analyze(req: Request) -> dict:
    """On-demand strategic read for a set of signals.

    Stakeholder doctrine (2026-04-29): the morning brief is FACTUAL pointers.
    Analysis happens on demand when the MD clicks "Analyze ↗" on a bullet
    or a bucket. This endpoint runs that analysis fresh on every call —
    no cache (stakeholder lock). Cost ~$0.001/call on the fast model.

    Body: {signal_ids: [int], scope: "bullet"|"bucket"|"section" (optional)}
    """
    from tools.nvidia_llm import chat_json

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")

    raw_ids = body.get("signal_ids") or []
    if not isinstance(raw_ids, list):
        raise HTTPException(status_code=400, detail="signal_ids must be a list")
    try:
        signal_ids = sorted({int(x) for x in raw_ids if str(x).strip()})
    except Exception:
        raise HTTPException(status_code=400, detail="signal_ids must be integers")
    if not signal_ids:
        raise HTTPException(status_code=400, detail="signal_ids is empty")
    if len(signal_ids) > 25:
        raise HTTPException(status_code=400, detail="signal_ids must be ≤ 25")

    scope = (body.get("scope") or "bullet").lower()

    # Pull the full signal rows so the LLM gets headline + dek + entities,
    # not just the IDs. One query, deterministic order.
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,decision_tag,"
                    "dev_slug,region,country_code,last_seen_at,urls,corroboration_count")
            .in_("id", signal_ids)
            .execute()
            .data
            or []
        )
    except Exception as e:
        log.warning("analyze: signal fetch failed: %s", e)
        raise HTTPException(status_code=502, detail="signal fetch failed")
    if not rows:
        raise HTTPException(status_code=404, detail="no signals matched the given ids")

    # Sobha context — gives the model the anchor / KPI lens. Fail-soft.
    try:
        portfolio = sobha_context.format_for_llm("summary")
    except Exception:
        portfolio = ""

    compact_signals = [
        {
            "id":         r.get("id"),
            "headline":   r.get("headline"),
            "dek":        r.get("dek"),
            "priority":   r.get("priority"),
            "category":   r.get("category"),
            "dev_slug":   r.get("dev_slug"),
            "region":     r.get("region"),
            "country":    r.get("country_code"),
            "last_seen":  r.get("last_seen_at"),
        }
        for r in rows
    ]
    user_payload = {
        "scope":             scope,
        "portfolio_summary": portfolio,
        "signals":           compact_signals,
    }

    # role="default" routes to the reasoning model (120B) — needed because
    # multi-signal analysis payloads can hit token limits on the 20B fast
    # model. Single-signal calls would still work on fast, but 2-25 signals
    # need the headroom. Cost is still <$0.01/call.
    try:
        parsed = await chat_json(
            [
                {"role": "system", "content": ANALYZE_SYSTEM},
                {"role": "user",   "content": _json.dumps(user_payload)},
            ],
            max_tokens=900,
            role="default",
        )
    except Exception as e:
        log.warning("analyze: LLM call failed: %s", e)
        raise HTTPException(status_code=502, detail="analysis unavailable")

    # Normalize the bullets shape — accept legacy {text, signal_ids} OR plain
    # strings (some models drop the object wrapper). Either way return the
    # frontend-expected {text, signal_ids} shape so formatBulletsHTML works.
    raw_bullets = parsed.get("bullets") or []
    bullets: list[dict[str, Any]] = []
    for b in raw_bullets:
        if isinstance(b, dict):
            text = str(b.get("text") or "").strip()
            ids = b.get("signal_ids") or []
            try:
                ids = [int(x) for x in ids if str(x).strip()]
            except Exception:
                ids = []
            if text:
                bullets.append({"text": text, "signal_ids": ids or signal_ids})
        elif isinstance(b, str) and b.strip():
            bullets.append({"text": b.strip(), "signal_ids": signal_ids})

    # Phase F2: ship the url index so the dashboard's Analyze panel
    # renders inline ↗ source chips on each strategic-read bullet.
    bullet_ids: set[int] = set(signal_ids)
    for b in bullets:
        for sid in (b.get("signal_ids") or []):
            try:
                bullet_ids.add(int(sid))
            except Exception:
                pass

    data_gaps      = [str(g).strip() for g in (parsed.get("data_gaps") or []) if str(g).strip()][:3]
    counterfactuals = [str(c).strip() for c in (parsed.get("counterfactuals") or []) if str(c).strip()][:3]
    stance         = str(parsed.get("stance") or "watch").lower()

    # Phase N5 (2026-04-29): persist every Analyze click so the MD's
    # Saved-Analyses rail can re-open a past read later. Best-effort —
    # never blocks the response.
    analysis_id: int | None = None
    try:
        # Derive a friendly label from the body's bucket / thread anchor.
        label = None
        body_bucket = body.get("bucket") or body.get("anchor")
        if isinstance(body_bucket, str) and body_bucket.strip():
            label = body_bucket.strip()[:80]
        else:
            label = (scope[:1].upper() + scope[1:]) + f" · {len(rows)} signals"
        ins = (
            sb.client().table("analyses")
            .insert({
                "user_id":      "md",
                "scope":        scope,
                "signal_ids":   signal_ids,
                "bullets":      bullets,
                "data_gaps":    data_gaps,
                "stance":       stance,
                "signal_count": len(rows),
                "label":        label,
            }).execute().data or []
        )
        if ins:
            analysis_id = ins[0].get("id")
    except Exception as e:
        log.info("analyze: persist failed: %s", e)

    return {
        "as_of":       datetime.now(timezone.utc).isoformat(),
        "scope":       scope,
        "signal_ids":  signal_ids,
        "bullets":     bullets,
        "data_gaps":   data_gaps,
        "counterfactuals": counterfactuals,
        "stance":      stance,
        "signal_count": len(rows),
        "signal_url_index": _build_signal_url_index(list(bullet_ids)),
        "analysis_id": analysis_id,
    }


@app.get("/api/analyses")
async def list_analyses(limit: int = 20) -> dict:
    """Saved Analyses rail (Phase N5).

    Returns the MD's recent Analyze-click results, freshest first. For each,
    we count how many NEW signals have landed against the same signal_ids
    since the analysis was created — the "what's new since" cue per card.
    """
    if limit < 1 or limit > 100:
        limit = 20
    try:
        rows = (
            sb.client().table("analyses")
            .select("id,scope,signal_ids,bullets,data_gaps,stance,signal_count,label,created_at")
            .eq("user_id", "md").is_("archived_at", "null")
            .order("created_at", desc=True).limit(limit).execute().data or []
        )
    except Exception as e:
        log.info("analyses list failed: %s", e)
        return {"items": []}

    # For each saved analysis, count related signal activity since it was created.
    out = []
    for a in rows:
        sids = a.get("signal_ids") or []
        sids = [int(x) for x in sids if isinstance(x, (int, str)) and str(x).isdigit()]
        new_since = 0
        if sids and a.get("created_at"):
            try:
                # Same dev_slug as any analyzed signal? Count.
                anchors = (
                    sb.client().table("signals")
                    .select("dev_slug").in_("id", sids).execute().data or []
                )
                devs = list({(r.get("dev_slug") or "").lower() for r in anchors if r.get("dev_slug")})
                if devs:
                    cnt_resp = (
                        sb.client().table("signals")
                        .select("id", count="exact")
                        .in_("dev_slug", devs)
                        .gte("last_seen_at", a["created_at"])
                        .not_.in_("id", sids)
                        .eq("archived", False).limit(1).execute()
                    )
                    new_since = cnt_resp.count or 0
            except Exception as e:
                log.info("analyses delta failed: %s", e)
        out.append({
            "id":            a.get("id"),
            "scope":         a.get("scope"),
            "label":         a.get("label"),
            "signal_count":  a.get("signal_count"),
            "stance":        a.get("stance"),
            "bullets":       a.get("bullets"),
            "data_gaps":     a.get("data_gaps"),
            "signal_ids":    sids,
            "created_at":    a.get("created_at"),
            "new_since":     new_since,
        })
    return {"items": out, "as_of": datetime.now(timezone.utc).isoformat()}


# 2026-04-29 Phase N1: tracking overview — velocity per bucket + delta-since-last-visit.
# Single round-trip so the dashboard banner + every synth-cell velocity strip
# render off one fetch on page load. No new tables — pure derivation from
# signals.last_seen_at + a category-bucket router that mirrors the brief.

# Same router as flows/daily_brief_flow.py:_categorize, kept inline here
# so the endpoint doesn't import from a circular path. If the canonical
# router changes, mirror the bucket vocabulary here.
_TRACKING_BUCKETS = ("markets", "construction", "tech", "regulatory",
                     "competitors", "dubai", "global")


def _bucket_for_signal(s: dict) -> str:
    cat = (s.get("category") or "").lower()
    dec = (s.get("decision_tag") or "").lower()
    dev = (s.get("dev_slug") or "").lower()
    region = (s.get("region") or "").lower()
    headline = (s.get("headline") or "").lower()
    # Direct category mapping first — most signals route deterministically.
    if cat in ("materials",) or "rebar" in headline or "shipping rate" in headline:
        return "construction"
    if cat in ("tech",) or "proptech" in headline or "contech" in headline:
        return "tech"
    if cat in ("capital_markets", "capital", "global_prime", "geopolitics", "capital_flow") \
       or dec == "capital":
        if cat in ("global_prime", "geopolitics", "capital_flow") or region == "other":
            return "global"
        return "markets"
    if cat in ("regulator",) or dec == "regulatory":
        return "regulatory"
    if dev in ("emaar", "damac", "aldar", "nakheel", "binghatti", "azizi",
               "meraas", "omniyat", "ellington", "deyaar", "modon", "dubaih"):
        return "competitors"
    if region == "other":
        return "global"
    return "dubai"


@app.get("/api/dashboard-tracking")
async def dashboard_tracking(since: str | None = None) -> dict:
    """Tracking overview powering the MD Scan delta banner + velocity strip.

    Query params:
      since (optional ISO8601): return count of new signals after this ts.
        Frontend passes localStorage `mdi_last_visit`. Omit on first visit.

    Returns:
      now:        current server time (ISO8601). Frontend stores this as
                  the new last_visit on banner-dismiss.
      since:      echoed back, or null.
      delta:      {total: N, by_bucket: {markets: 2, tech: 5, ...}}.
                  Empty when `since` omitted or no new signals.
      velocity:   {bucket: {today_per_day, prev_per_day, direction}} —
                  last-7d daily rate vs prior-7d. direction is one of:
                    'up'   (>= +50% WoW)
                    'flat' (-50%..+50%)
                    'down' (<= -50% WoW)
                  Stakeholder lock 2026-04-29: ±50% threshold, tunable later.
    """
    out: dict = {
        "now":      datetime.now(timezone.utc).isoformat(),
        "since":    since,
        "delta":    {"total": 0, "by_bucket": {b: 0 for b in _TRACKING_BUCKETS}},
        "velocity": {b: {"today_per_day": 0.0, "prev_per_day": 0.0, "direction": "flat"}
                     for b in _TRACKING_BUCKETS},
        # Phase N1.5 (2026-04-29): quiet-but-important detector. Inverse
        # of the velocity surge — flag buckets that are unusually QUIET
        # vs their 30-day baseline. Regulatory silence before a crackdown,
        # capital-markets calm before a sukuk window, etc.
        "quiet_buckets": [],
    }
    from datetime import timedelta as _td
    now_dt = datetime.now(timezone.utc)
    fourteen_d = (now_dt - _td(days=14)).isoformat()

    # One query: pull last 14d of signals (id + headline + category +
    # decision_tag + dev_slug + region + last_seen_at). Bucket client-side.
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,category,decision_tag,dev_slug,region,last_seen_at")
            .gte("last_seen_at", fourteen_d)
            .eq("archived", False)
            .order("last_seen_at", desc=True)
            .limit(2000)
            .execute().data or []
        )
    except Exception as e:
        log.info("dashboard-tracking: signals fetch failed: %s", e)
        return out

    # Velocity: count per-bucket in [now-7d, now] vs [now-14d, now-7d].
    seven_d_ago = now_dt - _td(days=7)
    counts_recent: dict[str, int] = {b: 0 for b in _TRACKING_BUCKETS}
    counts_prev:   dict[str, int] = {b: 0 for b in _TRACKING_BUCKETS}
    for r in rows:
        b = _bucket_for_signal(r)
        if b not in counts_recent:
            continue
        try:
            ts = datetime.fromisoformat(str(r.get("last_seen_at") or "").rstrip("Z"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts >= seven_d_ago:
            counts_recent[b] += 1
        else:
            counts_prev[b] += 1

    for b in _TRACKING_BUCKETS:
        rec = counts_recent[b] / 7.0
        prv = counts_prev[b]   / 7.0
        # Direction with the stakeholder-locked ±50% WoW threshold.
        if prv == 0 and rec > 0:
            direction = "up"
        elif rec == 0 and prv > 0:
            direction = "down"
        elif prv == 0 and rec == 0:
            direction = "flat"
        else:
            change = (rec - prv) / prv
            if change >= 0.50:
                direction = "up"
            elif change <= -0.50:
                direction = "down"
            else:
                direction = "flat"
        out["velocity"][b] = {
            "today_per_day": round(rec, 1),
            "prev_per_day":  round(prv, 1),
            "direction":     direction,
        }

    # ── Quiet-but-important detector ──
    # A bucket counts as "quiet" if its current 7d rate is BOTH:
    #   - ≤30% of the prior-7d rate (significant drop), AND
    #   - ≤50% of the trailing 30d baseline rate.
    # Two-condition gate prevents false-positives from buckets that
    # are always sparse (e.g. competitors weeks).
    # Why this matters: long calm in 'regulatory' before a DLD/RERA
    # decree, or in 'capital_markets' before a sukuk window, often
    # precedes a high-impact event.
    thirty_d_ago = now_dt - _td(days=30)
    counts_baseline: dict[str, int] = {b: 0 for b in _TRACKING_BUCKETS}
    for r in rows:
        b = _bucket_for_signal(r)
        if b not in counts_baseline:
            continue
        try:
            ts = datetime.fromisoformat(str(r.get("last_seen_at") or "").rstrip("Z"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts >= thirty_d_ago and ts < seven_d_ago:
            # Just the 23-day window before the recent 7d.
            counts_baseline[b] += 1
    for b in _TRACKING_BUCKETS:
        rec_per_day = counts_recent[b] / 7.0
        prv_per_day = counts_prev[b]   / 7.0
        baseline_per_day = counts_baseline[b] / 23.0
        # Gate: either baseline must be > 0.5 signals/day (otherwise
        # the bucket is structurally sparse and quiet-status is noise).
        if baseline_per_day < 0.5:
            continue
        # Two-condition: 7d ≤ 30% of prev-7d AND ≤ 50% of 30d baseline.
        if prv_per_day == 0:
            continue
        if rec_per_day > 0.3 * prv_per_day:
            continue
        if rec_per_day > 0.5 * baseline_per_day:
            continue
        out["quiet_buckets"].append({
            "bucket":             b,
            "today_per_day":      round(rec_per_day, 2),
            "baseline_per_day":   round(baseline_per_day, 2),
            "drop_vs_baseline":   round((1.0 - rec_per_day / baseline_per_day) * 100.0, 0),
        })

    # Delta since last visit.
    if since:
        try:
            since_dt = datetime.fromisoformat(since.rstrip("Z"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
            for r in rows:
                try:
                    ts = datetime.fromisoformat(str(r.get("last_seen_at") or "").rstrip("Z"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if ts > since_dt:
                    out["delta"]["total"] += 1
                    b = _bucket_for_signal(r)
                    if b in out["delta"]["by_bucket"]:
                        out["delta"]["by_bucket"][b] += 1
        except Exception as e:
            log.info("dashboard-tracking: bad since param '%s': %s", since, e)

    return out


## Project-name + move-type extractors for tighter thread clustering
## (Round-2, 2026-04-29). Heuristic — runs at thread-build time so we
## don't need to re-classify every signal. Move-type taxonomy aligns
## with the parked decision-readiness flag work.

# Sobha-relevant communities + headline patterns we recognise as projects.
_PROJECT_PATTERNS = [
    # Sobha
    "sobha hartland", "sobha one", "sobha seahaven", "sobha skyvue",
    "sobha verde", "sobha orbis", "sobha elwood", "sobha solis",
    "sobha reserve", "sobha realty",
    # Emaar flagship + areas
    "emaar beachfront", "creek harbour", "downtown dubai", "dubai hills",
    "emaar south", "arabian ranches", "expo living", "terra heights",
    # DAMAC
    "damac lagoons", "damac hills", "akoya", "dam ac islands",
    # Aldar
    "yas island", "saadiyat", "al fahid", "reem island",
    # Nakheel
    "palm jumeirah", "palm jebel ali", "deira islands", "the world",
    # Binghatti
    "binghatti mercedes", "mercedes places",
    # Modon / Reem
    "alma", "khalifa city",
    # Generic Dubai sub-markets
    "business bay", "dubai marina", "jumeirah village", "jvc", "jvt",
    "mbr city", "meydan", "dubailand", "al furjan", "tilal al ghaf",
    "maritime city",
]


# Move-type heuristics: scan headline + dek for these keyword groups.
_MOVE_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("pre_launch",  ("pre-launch", "prelaunch", "pre launch", "eoi", "expression of interest", "coming soon")),
    ("launch",      ("launch", "launches", "launched", "unveils", "now selling", "open for sale")),
    ("handover",    ("handover", "handed over", "delivers", "delivered", "completion", "ready")),
    ("sold_out",    ("sold out", "sold-out", "fully booked", "100% sold")),
    ("acquisition", ("acquisition", "acquires", "acquired", "buys", "bought", "purchase")),
    ("partnership", ("partnership", "partners with", "joint venture", "jv with", "tie-up")),
    ("payment_plan", ("payment plan", "post-handover", "post handover", "60/40", "70/30", "5-year payment", "5 year payment")),
    ("exec_move",   ("appointed", "joins as", "named ceo", "resigned", "stepping down", "new chairman")),
    ("distress",    ("delay", "delayed", "cancelled", "cancellation", "stalled", "paused", "default", "lawsuit")),
    ("capital",     ("sukuk", "bond", "issuance", "rating", "dividend", "buyback", "ipo", "listing")),
    ("regulatory",  ("dld", "rera", "freehold", "visa", "tax", "decree", "law")),
]


def _extract_project_name(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    for proj in _PROJECT_PATTERNS:
        if proj in t:
            return proj.title()  # display-cased
    return None


def _extract_move_type(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    for label, terms in _MOVE_TYPE_RULES:
        if any(term in t for term in terms):
            return label
    return None


# ─── Derived signals (Round-3, 2026-04-29) — items 109-115 ──────────────
# Computed from existing data (signals + metrics_daily + bayut_scout +
# dld_projects_scout). When the underlying data isn't there, return
# {"data_gap": "..."} honestly rather than fabricating numbers.

@app.get("/api/derived/buyer-nationality")
async def derived_buyer_nationality(days: int = 90) -> dict:
    """Buyer nationality + source-country mix — item 109.

    Cross-references DLD-tagged signals with country_code on capital_flow
    + global_prime signals. Returns top countries by signal volume in
    the lookback window. Honest gap when signals lack country tagging.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("country_code,category,headline")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .in_("category", ["capital_flow", "global_prime", "demand"])
            .limit(500).execute().data or []
        )
    except Exception as e:
        return {"data_gap": f"signals query failed: {e}", "by_country": {}}
    by_country: dict[str, int] = {}
    for r in rows:
        c = (r.get("country_code") or "").upper()
        if not c or c == "AE":
            continue
        by_country[c] = by_country.get(c, 0) + 1
    return {
        "by_country": dict(sorted(by_country.items(), key=lambda kv: -kv[1])[:15]),
        "window_days": days,
        "n_signals":   len(rows),
        "data_gap":    None if rows else "No nationality-tagged signals in window",
        "as_of":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/derived/currency-of-purchase")
async def derived_currency_of_purchase() -> dict:
    """Currency-of-purchase shift — item 110.

    Reads metrics_daily for FX cross-rates that matter to Dubai HNWI
    inflow: USD/INR (India), USD/RUB (Russia), USD/CNY (China),
    USD/GBP (UK). Computes 30-day % change per pair as a buyer-currency
    strength proxy.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=45)).isoformat()
    pairs = ["yf_INR_X", "yf_RUB_X", "yf_CNY_X", "yf_GBP_X", "yf_AUDUSD_X"]
    out: dict[str, Any] = {}
    for key in pairs:
        try:
            rows = (
                sb.client().table("metrics_daily")
                .select("captured_at,value_num")
                .eq("metric_key", key)
                .gte("captured_at", since)
                .order("captured_at").execute().data or []
            )
        except Exception:
            rows = []
        if len(rows) < 2:
            out[key] = {"data_gap": "insufficient history (<2 datapoints)"}
            continue
        first = float(rows[0]["value_num"] or 0)
        last  = float(rows[-1]["value_num"] or 0)
        if not first:
            out[key] = {"data_gap": "zero anchor"}
            continue
        pct = (last - first) / first * 100.0
        out[key] = {
            "first_value":  first,
            "last_value":   last,
            "pct_change":   round(pct, 2),
            "n_datapoints": len(rows),
        }
    return {
        "by_pair":  out,
        "as_of":    datetime.now(timezone.utc).isoformat(),
        "data_gap": None if any(isinstance(v, dict) and "pct_change" in v for v in out.values())
                         else "No FX history yet — wait for next ingest cycles",
    }


@app.get("/api/derived/mortgage-availability")
async def derived_mortgage_availability(days: int = 30) -> dict:
    """Mortgage availability drift — item 111.

    Reads metrics_daily for EIBOR + recent capital_markets signals
    mentioning mortgage rates. The richer compute (per-bank rate-card
    scrape) is parked — uses what we have.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    eibor_history: list[dict] = []
    try:
        eibor_history = (
            sb.client().table("metrics_daily")
            .select("captured_at,value_num,raw_json")
            .ilike("metric_key", "%eibor%")
            .gte("captured_at", since)
            .order("captured_at").execute().data or []
        )
    except Exception:
        pass
    eibor_summary = None
    if len(eibor_history) >= 2:
        first = float(eibor_history[0]["value_num"] or 0)
        last  = float(eibor_history[-1]["value_num"] or 0)
        eibor_summary = {
            "first": first, "last": last,
            "delta_bps": round((last - first) * 100, 1),
            "n_datapoints": len(eibor_history),
        }
    # Recent mortgage-rate signals from CBUAE / banks.
    try:
        mortgage_signals = (
            sb.client().table("signals")
            .select("id,headline,dek,last_seen_at")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .or_("headline.ilike.%mortgage%,dek.ilike.%mortgage%,headline.ilike.%LTV%,dek.ilike.%LTV%")
            .order("last_seen_at", desc=True)
            .limit(10).execute().data or []
        )
    except Exception:
        mortgage_signals = []
    return {
        "eibor":            eibor_summary,
        "mortgage_signals": mortgage_signals[:5],
        "data_gap":         None if (eibor_summary or mortgage_signals)
                                else "No EIBOR history or mortgage signals yet",
        "as_of":            datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/derived/future-supply")
async def derived_future_supply() -> dict:
    """Future supply pipeline by community + handover year — item 112.

    Counts dld_projects_scout signals per community + per handover year
    in the next 36 months. Trakheesi permits add construction-stage
    visibility.
    """
    from datetime import timedelta as _td
    # Look at all active project signals — supply pipeline doesn't decay.
    since = (datetime.now(timezone.utc) - _td(days=180)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,dev_slug,region")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .or_("headline.ilike.%launch%,headline.ilike.%handover%,headline.ilike.%phase%,dek.ilike.%units%")
            .limit(500).execute().data or []
        )
    except Exception as e:
        return {"data_gap": f"query failed: {e}"}
    by_dev: dict[str, int] = {}
    by_region: dict[str, int] = {}
    for r in rows:
        d = (r.get("dev_slug") or "_unknown").lower()
        rg = (r.get("region") or "_unknown").lower()
        by_dev[d] = by_dev.get(d, 0) + 1
        by_region[rg] = by_region.get(rg, 0) + 1
    return {
        "by_developer": dict(sorted(by_dev.items(), key=lambda kv: -kv[1])[:15]),
        "by_region":    dict(sorted(by_region.items(), key=lambda kv: -kv[1])),
        "n_signals":    len(rows),
        "window_days":  180,
        "data_gap":     None if rows else "No launch/handover signals in window",
        "as_of":        datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/derived/cancellation-rate")
async def derived_cancellation_rate(days: int = 90) -> dict:
    """Cancellation + project-restructuring rate — item 113.

    Counts signals matching distress / cancel / pause / delay patterns
    over the lookback. By developer.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,dev_slug,last_seen_at")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .or_(
                "headline.ilike.%cancel%,headline.ilike.%delay%,"
                "headline.ilike.%paused%,headline.ilike.%stalled%,"
                "headline.ilike.%distress%,headline.ilike.%default%,"
                "dek.ilike.%cancel%,dek.ilike.%delay%,dek.ilike.%paused%,"
                "dek.ilike.%stalled%,dek.ilike.%distress%"
            )
            .order("last_seen_at", desc=True)
            .limit(200).execute().data or []
        )
    except Exception as e:
        return {"data_gap": f"query failed: {e}"}
    by_dev: dict[str, int] = {}
    for r in rows:
        d = (r.get("dev_slug") or "_unknown").lower()
        by_dev[d] = by_dev.get(d, 0) + 1
    return {
        "by_developer": dict(sorted(by_dev.items(), key=lambda kv: -kv[1])[:15]),
        "recent_examples": [
            {"id": r.get("id"), "headline": r.get("headline"),
             "dev": r.get("dev_slug"), "last_seen_at": r.get("last_seen_at")}
            for r in rows[:8]
        ],
        "n_signals":   len(rows),
        "window_days": days,
        "data_gap":    None if rows else "No distress/cancellation signals in window",
        "as_of":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/derived/land-only")
async def derived_land_only(days: int = 90) -> dict:
    """Land-only transaction prices — item 114.

    Filters signals matching land/plot/auction patterns. Combined with
    emart_auctions_scout's distress-land-auction feed.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,dev_slug,region,country_code,last_seen_at,urls")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .or_(
                "headline.ilike.%land plot%,headline.ilike.%land auction%,"
                "headline.ilike.%plot of land%,headline.ilike.%land bank%,"
                "headline.ilike.%land deal%,headline.ilike.%land acquisition%,"
                "dek.ilike.%land plot%,dek.ilike.%land auction%"
            )
            .order("last_seen_at", desc=True)
            .limit(50).execute().data or []
        )
    except Exception as e:
        return {"data_gap": f"query failed: {e}"}
    return {
        "transactions": rows[:25],
        "n_transactions": len(rows),
        "window_days":  days,
        "data_gap":     None if rows else "No land-only transactions surfaced",
        "as_of":        datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/derived/rental-yield")
async def derived_rental_yield(days: int = 90) -> dict:
    """Rental yield drift by segment — item 115.

    Counts rental-yield-mentioning signals + their direction. Real RERA
    Rental Index integration is parked — uses signal commentary today.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,region,country_code,last_seen_at")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .or_(
                "headline.ilike.%rental yield%,headline.ilike.%gross yield%,"
                "headline.ilike.%net yield%,headline.ilike.%cap rate%,"
                "headline.ilike.%rent index%,headline.ilike.%rental price%,"
                "dek.ilike.%rental yield%,dek.ilike.%cap rate%"
            )
            .order("last_seen_at", desc=True)
            .limit(50).execute().data or []
        )
    except Exception as e:
        return {"data_gap": f"query failed: {e}"}
    by_region: dict[str, int] = {}
    for r in rows:
        rg = (r.get("region") or "_unknown").lower()
        by_region[rg] = by_region.get(rg, 0) + 1
    return {
        "by_region":    by_region,
        "yield_signals": rows[:15],
        "n_signals":    len(rows),
        "window_days":  days,
        "data_gap":     None if rows else "No rental-yield signals in window — wait for RERA index integration",
        "as_of":        datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/active-threads")
async def active_threads(days: int = 14, min_signals: int = 3) -> dict:
    """Multi-day stories the system is following (Phase N2, 2026-04-29).

    A "thread" is a cluster of signals that share an anchor (developer
    or category-region) and have ≥`min_signals` distinct evidence points
    across the lookback window. Surfaces on MD Scan above the synth cells
    so the MD sees follow-through, not just freshness.

    Cheaper than embedding-based clustering and aligned with how the
    reader actually thinks ("Binghatti payment delays · 6 signals · 12 days").

    Returns:
      threads: [{
        anchor_kind: "developer" | "category",
        anchor: "binghatti" | "regulator:dubai",
        title: "Binghatti — 6 signals across 12 days",
        signal_count: int,
        span_days: int,        # first to last seen
        latest_at: ISO8601,
        sample_signal_ids: [int]  # 5 most-recent for drilling
      }, ...]
    """
    from datetime import timedelta as _td
    if days < 3 or days > 60:
        raise HTTPException(status_code=400, detail="days must be 3..60")
    if min_signals < 2 or min_signals > 20:
        raise HTTPException(status_code=400, detail="min_signals must be 2..20")
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dev_slug,category,region,last_seen_at,priority")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .order("last_seen_at", desc=True)
            .limit(2000)
            .execute().data or []
        )
    except Exception as e:
        log.info("active-threads: fetch failed: %s", e)
        return {"threads": []}

    # 2026-04-29 Round-2: tighter clustering — group by
    # (dev_slug, project_name, move_type) instead of just dev_slug.
    # The bigger Binghatti story might be 4 signals about Mercedes Places
    # AND 2 signals about another project; current logic conflates them.
    # New logic: tag every signal with project_name + move_type at thread
    # time using the heuristic extractors above, then group on the triple.
    # Fallback for null dev_slug: group by (category, region) as before.
    dev_groups: dict[str, list[dict]] = {}
    cat_groups: dict[str, list[dict]] = {}
    for r in rows:
        # Annotate the signal in-memory so thread builders can read these.
        text = ((r.get("headline") or "") + " " + (r.get("dek") or ""))
        r["_project_name"] = _extract_project_name(text)
        r["_move_type"]    = _extract_move_type(text)
        dev = (r.get("dev_slug") or "").lower()
        if dev:
            # Triple key: dev × project × move_type. Fallback when project
            # or move_type are unknown so we still get a thread for general
            # developer-level chatter.
            project = r["_project_name"] or "_general"
            move    = r["_move_type"]    or "_mixed"
            key = f"{dev}|{project}|{move}"
            dev_groups.setdefault(key, []).append(r)
        else:
            cat = (r.get("category") or "other").lower()
            region = (r.get("region") or "other").lower()
            cat_groups.setdefault(f"{cat}:{region}", []).append(r)

    threads: list[dict] = []
    DEV_LABEL = {
        "emaar": "Emaar", "damac": "DAMAC", "aldar": "Aldar", "nakheel": "Nakheel",
        "binghatti": "Binghatti", "azizi": "Azizi", "ellington": "Ellington",
        "omniyat": "Omniyat", "deyaar": "Deyaar", "modon": "Modon",
        "dubaih": "Dubai Holding", "meraas": "Meraas", "sobha": "Sobha",
    }

    def _build_thread(anchor_kind: str, anchor: str, sigs: list[dict], title_label: str) -> dict | None:
        if len(sigs) < min_signals:
            return None
        # Span: first to last seen. Compute via parsed timestamps.
        ts: list[datetime] = []
        for s in sigs:
            try:
                t = datetime.fromisoformat(str(s.get("last_seen_at") or "").rstrip("Z"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                ts.append(t)
            except Exception:
                pass
        if not ts:
            return None
        ts.sort()
        # Require ≥3 distinct calendar days for a real thread (otherwise
        # 5 signals all from one launch announcement would qualify).
        distinct_days = {t.date() for t in ts}
        if len(distinct_days) < min(3, min_signals):
            return None
        span_days = max(1, (ts[-1] - ts[0]).days + 1)
        latest_at = ts[-1].isoformat()
        n = len(sigs)
        return {
            "anchor_kind":       anchor_kind,
            "anchor":            anchor,
            "title":             f"{title_label} — {n} signals across {span_days} days",
            "signal_count":      n,
            "span_days":         span_days,
            "distinct_days":     len(distinct_days),
            "latest_at":         latest_at,
            "sample_signal_ids": [s.get("id") for s in sigs[:5] if s.get("id") is not None],
            "sample_headlines":  [s.get("headline") or "" for s in sigs[:3]],
        }

    for triple_key, sigs in dev_groups.items():
        # triple_key shape: "dev|project|move_type"
        parts = triple_key.split("|", 2)
        dev = parts[0]
        project = parts[1] if len(parts) > 1 else "_general"
        move    = parts[2] if len(parts) > 2 else "_mixed"
        # Friendly label: "Binghatti · Mercedes Places · launch"
        bits: list[str] = [DEV_LABEL.get(dev, dev.title())]
        if project and project != "_general":
            bits.append(project)
        if move and move != "_mixed":
            bits.append(move.replace("_", "-"))
        title_label = " · ".join(bits)
        t = _build_thread("developer", triple_key, sigs, title_label)
        if t:
            t["dev_slug"]     = dev
            t["project_name"] = None if project == "_general" else project
            t["move_type"]    = None if move == "_mixed" else move
            threads.append(t)
    for key, sigs in cat_groups.items():
        cat, region = key.split(":", 1)
        # Friendlier label.
        label = cat.replace("_", " ").title()
        if region and region != "other":
            label = f"{label} · {region.replace('_', ' ').title()}"
        t = _build_thread("category", key, sigs, label)
        if t:
            threads.append(t)

    # Rank: more signals first; break ties on more recent activity.
    threads.sort(key=lambda x: (x["signal_count"], x["latest_at"]), reverse=True)
    return {"threads": threads[:12], "as_of": datetime.now(timezone.utc).isoformat()}


@app.get("/api/story-arcs")
async def story_arcs(days: int = 14, min_signals: int = 4) -> dict:
    """Cross-bucket story arcs (Round-2 N-extension, 2026-04-29).

    Threads cluster within a developer; arcs cluster across BUCKETS.
    Example: "Iran-Israel escalation → oil spike → DSC CCI → material
    price impact → Sobha margin pressure" — that's 4-5 signals across
    4 different buckets (geopolitics, markets, materials, competitors).

    Approach: pull recent signals with their pgvector embeddings, run
    DBSCAN-lite clustering on cosine distance with a relatively loose
    similarity threshold (0.78). Filter to clusters that span ≥2
    distinct buckets with ≥`min_signals` signals total.

    Output:
      arcs: [{
        arc_id: hash,
        title: short summary
        signal_ids: [int]
        spans_buckets: [bucket]
        signal_count: int
        latest_at: ISO
        sample_headlines: [str]
      }]

    Compute is bounded: top 200 recent signals, O(n²) similarity comparison
    on the python side. Doesn't need a new RPC.
    """
    if days < 3 or days > 30:
        days = 14
    if min_signals < 3 or min_signals > 20:
        min_signals = 4

    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,category,decision_tag,dev_slug,region,"
                    "embedding,last_seen_at,priority")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .order("last_seen_at", desc=True)
            .limit(200).execute().data or []
        )
    except Exception as e:
        log.info("story-arcs: fetch failed: %s", e)
        return {"arcs": []}

    # Drop signals without embeddings — clustering needs them. Pre-2026
    # rows may be missing embeddings.
    eligible = []
    for r in rows:
        emb = r.get("embedding")
        if emb and isinstance(emb, list) and len(emb) >= 100:
            r["_emb"] = emb
            eligible.append(r)
    if len(eligible) < min_signals:
        return {"arcs": []}

    # ── DBSCAN-lite via cosine similarity ──
    # Threshold tuned conservatively so arcs don't over-merge unrelated
    # signals. Each signal seeds a cluster; we sweep the eligible list
    # and merge if cosine ≥ COS_T against any cluster member.
    COS_T = 0.78

    def _cos(a: list[float], b: list[float]) -> float:
        try:
            import math
            n = min(len(a), len(b))
            if n == 0:
                return 0.0
            dot = sum(a[i] * b[i] for i in range(n))
            na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
            nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)
        except Exception:
            return 0.0

    clusters: list[list[dict]] = []
    for sig in eligible:
        placed = False
        for cluster in clusters:
            # Compare against the centroid (mean of cluster member embeddings)
            # — fast enough at this scale.
            centroid = cluster[0]["_emb"]
            if _cos(sig["_emb"], centroid) >= COS_T:
                cluster.append(sig)
                placed = True
                break
        if not placed:
            clusters.append([sig])

    # Build arcs: keep clusters with ≥min_signals signals AND ≥2 distinct
    # buckets (otherwise it's just a single-bucket thread).
    arcs: list[dict] = []
    for c in clusters:
        if len(c) < min_signals:
            continue
        buckets = list({_bucket_for_signal(s) for s in c})
        if len(buckets) < 2:
            continue
        # Span + latest.
        ts: list[datetime] = []
        for s in c:
            try:
                t = datetime.fromisoformat(str(s.get("last_seen_at") or "").rstrip("Z"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                ts.append(t)
            except Exception:
                pass
        if not ts:
            continue
        ts.sort()
        # Title: shortest headline in cluster (often the most general).
        cluster_sorted = sorted(c, key=lambda s: len(s.get("headline") or ""))
        title_seed = cluster_sorted[0].get("headline") or ""
        signal_ids = [s.get("id") for s in c if s.get("id") is not None]
        arc_hash = hashlib.sha256(
            ("|".join(str(i) for i in sorted(signal_ids))).encode()
        ).hexdigest()[:16]
        arcs.append({
            "arc_id":           arc_hash,
            "title":            f"Story arc · spans {', '.join(sorted(buckets))} · {len(c)} signals across {(ts[-1]-ts[0]).days+1} days",
            "title_seed":       title_seed[:200],
            "signal_ids":       signal_ids,
            "spans_buckets":    sorted(buckets),
            "signal_count":     len(c),
            "span_days":        max(1, (ts[-1] - ts[0]).days + 1),
            "latest_at":        ts[-1].isoformat(),
            "sample_headlines": [s.get("headline") or "" for s in c[:3]],
        })
    arcs.sort(key=lambda a: (a["signal_count"], a["latest_at"]), reverse=True)
    return {
        "arcs":  arcs[:10],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/yesterday-pointers")
async def yesterday_pointers() -> dict:
    """Yesterday's brief top-signals re-shown with a status tag (Phase N3).

    Status taxonomy:
      'still-active' — a signal with the same dev_slug had new evidence
                       in the last 24h (i.e. the story is still moving).
      'resolved'     — a follow-up signal whose headline mentions
                       'completed' / 'closed' / 'sold out' / 'handover' /
                       'delivered' has landed in the last 24h.
      'quiet'        — no new evidence in 24h. The story has gone cold.

    Drives the MD Scan "Yesterday's pointers — what changed" rail so the
    MD sees follow-through, not just freshness.
    """
    from datetime import timedelta as _td
    today_dt = datetime.now(timezone.utc).date()
    # Yesterday's brief.
    try:
        rows = (
            sb.client().table("daily_briefs")
            .select("day,top_signals")
            .order("day", desc=True).limit(7).execute().data or []
        )
    except Exception as e:
        log.info("yesterday-pointers: brief fetch failed: %s", e)
        return {"pointers": []}
    # Pick the most-recent brief that's NOT today's — that's yesterday's
    # read; brief generation may straddle midnight so we tolerate up to
    # 3 days back.
    yesterday_brief = None
    for r in rows:
        try:
            d = datetime.fromisoformat(str(r.get("day"))).date()
        except Exception:
            continue
        if d < today_dt:
            yesterday_brief = r
            break
    if not yesterday_brief:
        return {"pointers": [], "note": "no prior brief found"}

    top = yesterday_brief.get("top_signals") or []
    if not top:
        return {"pointers": [], "note": "yesterday's brief had no top signals"}
    yesterday_ids = [t.get("signal_id") or t.get("id") for t in top if t.get("signal_id") or t.get("id")]
    if not yesterday_ids:
        return {"pointers": [], "note": "yesterday's top signals carried no ids"}

    # Pull each signal's full row to get dev_slug / category for follow-up matching.
    try:
        sig_rows = (
            sb.client().table("signals")
            .select("id,headline,dek,dev_slug,category,region,last_seen_at,urls,priority")
            .in_("id", yesterday_ids)
            .execute().data or []
        )
    except Exception as e:
        log.info("yesterday-pointers: signal fetch failed: %s", e)
        sig_rows = []
    sig_by_id = {s.get("id"): s for s in sig_rows}

    # Pull last 24h signals to test for follow-up activity.
    one_d_ago = (datetime.now(timezone.utc) - _td(hours=24)).isoformat()
    try:
        recent = (
            sb.client().table("signals")
            .select("id,headline,dev_slug,category,last_seen_at")
            .gte("last_seen_at", one_d_ago)
            .eq("archived", False)
            .order("last_seen_at", desc=True)
            .limit(500)
            .execute().data or []
        )
    except Exception as e:
        log.info("yesterday-pointers: recent fetch failed: %s", e)
        recent = []
    recent_by_dev: dict[str, list[dict]] = {}
    for r in recent:
        d = (r.get("dev_slug") or "").lower()
        if d:
            recent_by_dev.setdefault(d, []).append(r)

    RESOLVED_RX = re.compile(
        r"\b(complete[d]?|closed|sold[\s-]*out|handed over|handover|delivered|"
        r"cancel(?:ed|led)?|withdrawn|paused|terminated)\b",
        re.IGNORECASE,
    )

    pointers = []
    for t in top[:6]:
        sid = t.get("signal_id") or t.get("id")
        if not sid:
            continue
        s = sig_by_id.get(sid) or {}
        dev = (s.get("dev_slug") or t.get("dev_slug") or "").lower()
        followups = recent_by_dev.get(dev, []) if dev else []
        # Filter: a follow-up must be a DIFFERENT signal id (not the
        # yesterday-pointer itself recurring), AND on the same dev.
        followups = [f for f in followups if f.get("id") != sid]
        # Resolution test on follow-up headlines.
        resolved_evidence = next((f for f in followups
                                 if RESOLVED_RX.search(f.get("headline") or "")), None)
        if resolved_evidence:
            status = "resolved"
            evidence_signal_id = resolved_evidence.get("id")
            evidence_headline = resolved_evidence.get("headline")
        elif followups:
            status = "still-active"
            evidence_signal_id = followups[0].get("id")
            evidence_headline = followups[0].get("headline")
        else:
            status = "quiet"
            evidence_signal_id = None
            evidence_headline = None
        pointers.append({
            "signal_id":          sid,
            "headline":           t.get("headline") or s.get("headline") or "",
            "dek":                t.get("dek") or s.get("dek") or "",
            "dev_slug":           dev or None,
            "url":                t.get("url") or ((s.get("urls") or [None])[0]),
            "status":             status,  # still-active | resolved | quiet
            "followup_count":     len(followups),
            "evidence_signal_id": evidence_signal_id,
            "evidence_headline":  evidence_headline,
        })
    return {
        "pointers":  pointers,
        "for_day":   yesterday_brief.get("day"),
        "as_of":     datetime.now(timezone.utc).isoformat(),
    }


# ── N4 (Phase N4, 2026-04-29): Watchlist — pinned signals/patterns/threads/analyses.
# Single-user (user_id="md") for now. Persists across devices via Supabase.

@app.post("/api/watchlist")
async def watchlist_pin(req: Request) -> dict:
    """Pin a signal / pattern / thread / analysis to the MD's watchlist.

    Body: {target_kind, target_id, label?, note?}
    Re-pinning the same item un-dismisses it (idempotent toggle-back).
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    target_kind = (body.get("target_kind") or "").lower()
    if target_kind not in ("signal", "pattern", "thread", "analysis"):
        raise HTTPException(status_code=400, detail="target_kind must be one of: signal, pattern, thread, analysis")
    target_id = str(body.get("target_id") or "").strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="target_id required")
    payload = {
        "user_id":      "md",
        "target_kind":  target_kind,
        "target_id":    target_id,
        "label":        (body.get("label") or "").strip()[:200] or None,
        "note":         (body.get("note")  or "").strip()[:500] or None,
        "dismissed_at": None,
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        # Upsert on (user_id, target_kind, target_id) — re-pin clears dismissed_at.
        result = (
            sb.client().table("watchlist")
            .upsert(payload, on_conflict="user_id,target_kind,target_id")
            .execute().data or []
        )
    except Exception as e:
        log.warning("watchlist pin failed: %s", e)
        raise HTTPException(status_code=502, detail="pin failed")
    return {"ok": True, "row": result[0] if result else payload}


@app.delete("/api/watchlist/{item_id}")
async def watchlist_unpin(item_id: int) -> dict:
    """Soft-unpin: set dismissed_at = now(). Row stays for audit."""
    try:
        result = (
            sb.client().table("watchlist")
            .update({"dismissed_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", item_id).execute().data or []
        )
    except Exception as e:
        log.warning("watchlist unpin failed: %s", e)
        raise HTTPException(status_code=502, detail="unpin failed")
    if not result:
        raise HTTPException(status_code=404, detail="watchlist item not found")
    return {"ok": True}


@app.get("/api/watchlist")
async def watchlist_list() -> dict:
    """Return active (un-dismissed) watchlist items + their latest evidence.

    For each pinned signal/pattern, we stitch in:
      - the latest follow-up signal headline + last_seen_at (signal only)
      - signal_url_index for any inline ↗ rendering
    """
    try:
        rows = (
            sb.client().table("watchlist")
            .select("id,target_kind,target_id,label,note,pinned_at,last_seen_at")
            .eq("user_id", "md")
            .is_("dismissed_at", "null")
            .order("pinned_at", desc=True)
            .limit(40)
            .execute().data or []
        )
    except Exception as e:
        log.info("watchlist list failed: %s", e)
        return {"items": []}

    # Enrich pinned signals with latest follow-up + URL index.
    signal_ids: list[int] = []
    for r in rows:
        if r.get("target_kind") == "signal":
            try:
                signal_ids.append(int(r["target_id"]))
            except Exception:
                pass
    sig_by_id: dict[int, dict] = {}
    if signal_ids:
        try:
            srows = (
                sb.client().table("signals")
                .select("id,headline,dek,dev_slug,category,region,last_seen_at,urls,priority")
                .in_("id", signal_ids).execute().data or []
            )
            sig_by_id = {s["id"]: s for s in srows if s.get("id") is not None}
        except Exception as e:
            log.info("watchlist sig enrich failed: %s", e)

    # Pull last 7d follow-ups for each pinned signal's dev_slug (the "what's
    # changed since pin" cue on each card).
    from datetime import timedelta as _td
    seven_d = (datetime.now(timezone.utc) - _td(days=7)).isoformat()
    devs = {(s.get("dev_slug") or "").lower() for s in sig_by_id.values()}
    devs.discard("")
    followups_by_dev: dict[str, dict] = {}
    if devs:
        try:
            recent = (
                sb.client().table("signals")
                .select("id,headline,dev_slug,last_seen_at")
                .in_("dev_slug", list(devs))
                .gte("last_seen_at", seven_d)
                .order("last_seen_at", desc=True)
                .limit(200).execute().data or []
            )
            for r in recent:
                d = (r.get("dev_slug") or "").lower()
                if d and d not in followups_by_dev:
                    # First (most-recent) follow-up per dev that isn't the
                    # pinned signal itself. Caller filters self-matches.
                    followups_by_dev[d] = r
        except Exception as e:
            log.info("watchlist followup fetch failed: %s", e)

    enriched = []
    for r in rows:
        item = {
            "id":           r.get("id"),
            "target_kind":  r.get("target_kind"),
            "target_id":    r.get("target_id"),
            "label":        r.get("label"),
            "note":         r.get("note"),
            "pinned_at":    r.get("pinned_at"),
            "last_seen_at": r.get("last_seen_at"),
        }
        if r.get("target_kind") == "signal":
            try:
                sid = int(r["target_id"])
            except Exception:
                sid = None
            s = sig_by_id.get(sid) if sid else None
            if s:
                item["signal"] = {
                    "id":           s.get("id"),
                    "headline":     s.get("headline"),
                    "dek":          s.get("dek"),
                    "dev_slug":     s.get("dev_slug"),
                    "category":     s.get("category"),
                    "url":          (s.get("urls") or [None])[0],
                    "last_seen_at": s.get("last_seen_at"),
                    "priority":     s.get("priority"),
                }
                # Follow-up evidence — same dev, different signal id.
                dev = (s.get("dev_slug") or "").lower()
                fu = followups_by_dev.get(dev) if dev else None
                if fu and fu.get("id") != s.get("id"):
                    item["latest_followup"] = {
                        "id":            fu.get("id"),
                        "headline":      fu.get("headline"),
                        "last_seen_at": fu.get("last_seen_at"),
                    }
        enriched.append(item)
    return {
        "items":  enriched,
        "as_of":  datetime.now(timezone.utc).isoformat(),
    }


# 2026-04-29 Round-2: telemetry ingest. Frontend posts here when the MD
# clicks Analyze, pins/unpins, opens a thread, etc. Closes the loop —
# the classifier-priority feedback work reads from here to learn implicit
# relevance ("MD opens Binghatti threads twice as often as Aldar threads").

_VALID_TELEMETRY_EVENTS = {
    "analyze_click", "pin", "unpin", "thread_open",
    "bullet_source_click", "signal_source_click",
    "page_view", "banner_dismiss",
}


@app.post("/api/telemetry")
async def telemetry_ingest(req: Request) -> dict:
    """Log a single MD interaction event. Body:
        {event, target_kind?, target_id?, context?}
    Best-effort — never raises. Used by the dashboard's click handlers.
    """
    try:
        body = await req.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}
    event = (body.get("event") or "").strip()
    if event not in _VALID_TELEMETRY_EVENTS:
        return {"ok": False, "error": f"invalid event '{event}'"}
    payload = {
        "user_id":     "md",
        "event":       event,
        "target_kind": (body.get("target_kind") or None),
        "target_id":   (str(body.get("target_id")) if body.get("target_id") is not None else None),
        "context":     body.get("context") or {},
    }
    try:
        client_resp = sb.client().table("telemetry").insert(payload).execute()
        return {"ok": True, "id": (client_resp.data or [{}])[0].get("id")}
    except Exception as e:
        log.info("telemetry insert failed: %s", e)
        return {"ok": False, "error": "persist failed"}


@app.get("/api/telemetry/feedback-weights")
async def telemetry_feedback_weights(days: int = 30) -> dict:
    """Closes the loop: produces per-(dev_slug + category) implicit relevance
    weights from MD telemetry (Phase N5+, 2026-04-29).

    Algorithm:
      - Pull last `days` of telemetry events
      - For each event with a target signal/thread/analysis, dereference
        to the underlying signal(s) and tally by (dev_slug, category)
      - Normalize so the highest-engaged combo = 1.0; floor at 0.0
      - Engagement weights: pin = 3, analyze_click = 2, thread_open = 1,
        source_click = 0.5

    Used by the brief and ceo_scan flows to BUMP priority on signals
    that match high-weight (dev, category) pairs. Returns the raw
    weights so the consuming flow can decide how to apply them.
    """
    if days < 7 or days > 90:
        days = 30
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        events = (
            sb.client().table("telemetry")
            .select("event,target_kind,target_id")
            .gte("created_at", since)
            .eq("user_id", "md")
            .limit(2000).execute().data or []
        )
    except Exception:
        return {"weights": {}, "events_seen": 0}

    EVENT_WEIGHTS = {
        "pin": 3.0,
        "analyze_click": 2.0,
        "thread_open": 1.0,
        "bullet_source_click": 0.5,
        "signal_source_click": 0.5,
    }

    # Walk targets → underlying signals.
    signal_id_targets: list[tuple[int, float]] = []
    for ev in events:
        kind = ev.get("target_kind")
        tid  = ev.get("target_id")
        w    = EVENT_WEIGHTS.get(ev.get("event"), 0.0)
        if w <= 0 or not tid:
            continue
        if kind == "signal":
            try:
                signal_id_targets.append((int(tid), w))
            except Exception:
                pass
        # thread / analysis → resolve later if we want; for now skip.

    if not signal_id_targets:
        return {"weights": {}, "events_seen": len(events)}

    # Pull (dev_slug, category) for all referenced signals.
    sids = list({sid for sid, _ in signal_id_targets})
    try:
        sigs = (
            sb.client().table("signals")
            .select("id,dev_slug,category")
            .in_("id", sids).execute().data or []
        )
    except Exception:
        return {"weights": {}, "events_seen": len(events)}
    by_id = {s["id"]: s for s in sigs}

    # Tally weights per (dev_slug, category).
    raw: dict[str, float] = {}
    for sid, w in signal_id_targets:
        s = by_id.get(sid)
        if not s:
            continue
        dev = (s.get("dev_slug") or "_none").lower()
        cat = (s.get("category") or "_none").lower()
        key = f"{dev}|{cat}"
        raw[key] = raw.get(key, 0.0) + w

    if not raw:
        return {"weights": {}, "events_seen": len(events)}

    # Normalise to [0, 1].
    max_w = max(raw.values())
    weights = {k: round(v / max_w, 3) for k, v in raw.items()}
    # Sort descending for nice JSON.
    weights = dict(sorted(weights.items(), key=lambda kv: -kv[1]))
    return {
        "weights":     weights,
        "events_seen": len(events),
        "window_days": days,
        "as_of":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/telemetry/rollup")
async def telemetry_rollup(days: int = 7) -> dict:
    """Per-target click rollup over the last N days. Used by the
    MD-feedback loop to surface 'most-engaged' targets back into the
    classifier's priority weighting.

    Returns: {by_target: [{target_kind, target_id, event_count, last_at}]}
    """
    if days < 1 or days > 90:
        days = 7
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("telemetry")
            .select("event,target_kind,target_id,created_at")
            .gte("created_at", since)
            .eq("user_id", "md")
            .order("created_at", desc=True)
            .limit(2000).execute().data or []
        )
    except Exception as e:
        log.info("telemetry rollup failed: %s", e)
        return {"by_target": [], "by_event": {}}
    by_event: dict[str, int] = {}
    by_target: dict[str, dict[str, Any]] = {}
    for r in rows:
        ev = r.get("event") or "—"
        by_event[ev] = by_event.get(ev, 0) + 1
        kind = r.get("target_kind")
        tid  = r.get("target_id")
        if not kind or tid is None:
            continue
        key = f"{kind}:{tid}"
        if key not in by_target:
            by_target[key] = {
                "target_kind": kind, "target_id": tid,
                "event_count": 0, "last_at": r.get("created_at"),
            }
        by_target[key]["event_count"] += 1
    return {
        "by_event":  dict(sorted(by_event.items(),  key=lambda kv: -kv[1])),
        "by_target": sorted(by_target.values(), key=lambda x: -x["event_count"])[:30],
        "as_of":     datetime.now(timezone.utc).isoformat(),
        "window_days": days,
    }


@app.get("/admin/recent-signals")
async def admin_recent_signals(
    hours: int = 24, limit: int = 100, category: str = "", source: str = "",
    authorization: str | None = Header(default=None),
) -> dict:
    _require_admin(authorization)
    """Debugging surface for classifier routing.

    Returns recent signals with their classifier-assigned category +
    decision_tag + source so we can verify the new category enum
    (tech / materials / geopolitics / global_prime / capital_flow) is
    actually being applied vs. legacy categories. Filterable.
    """
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(hours=hours)).isoformat()
    # 2026-04-29 fix: signals table doesn't have a source_kind column
    # (we use category_hint at promotion instead — see ingest_flow keystone).
    # The "source" of a signal is inferred from urls[0] domain. Adapt the
    # admin view to that reality.
    try:
        q = (
            sb.client().table("signals")
            .select("id,headline,category,decision_tag,priority,region,"
                    "country_code,dev_slug,last_seen_at,source_published_at,"
                    "urls,corroboration_count,corroborating_sources")
            .gte("last_seen_at", since)
            .eq("archived", False)
            .order("last_seen_at", desc=True)
            .limit(limit)
        )
        rows = q.execute().data or []
    except Exception as e:
        return {"error": str(e), "signals": []}
    # Derive a friendly source-host string from urls[0] for the rollup.
    from urllib.parse import urlparse as _urlparse
    for r in rows:
        urls = r.get("urls") or []
        host = ""
        if urls:
            try:
                host = _urlparse(urls[0]).netloc.lower().replace("www.", "")
            except Exception:
                pass
        r["_source_host"] = host or "—"
    if category:
        rows = [r for r in rows if (r.get("category") or "") == category]
    if source:
        rows = [r for r in rows if source.lower() in r["_source_host"]]
    # Tallies.
    by_cat: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in rows:
        c = (r.get("category") or "—")
        by_cat[c] = by_cat.get(c, 0) + 1
        by_source[r["_source_host"]] = by_source.get(r["_source_host"], 0) + 1
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "hours": hours,
        "count": len(rows),
        "by_category":   dict(sorted(by_cat.items(),   key=lambda kv: -kv[1])),
        "by_source_host": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
        "signals": rows,
    }


@app.get("/admin/sobha-portfolio")
async def get_sobha_portfolio(
    authorization: str | None = Header(default=None),
) -> dict:
    """Inspect what Sobha context is currently loaded.

    Returns the same digest LLM stages see via format_for_llm("summary").
    Useful for verifying a sync worked and for the dashboard to render
    a small "context loaded" indicator.
    """
    _require_admin(authorization)
    return {
        "summary": sobha_context.get_portfolio_summary(),
        "preamble": sobha_context.PERSONA_PREAMBLE,
        "summary_text": sobha_context.format_for_llm("summary"),
    }


@app.get("/admin/sobha-overlap")
async def get_sobha_overlap_endpoint(
    dev: str = "",
    authorization: str | None = Header(default=None),
) -> dict:
    """Sobha's competitive overlap with a named developer (used by Friday).

    Returns regions where both operate, community overlap, adjacent
    Sobha projects, segment overlap, and analyst notes. Falls back to
    a "not mapped" payload when the developer isn't in the curated map.
    """
    _require_admin(authorization)
    overlap = sobha_context.get_competitive_overlap(dev or None)
    return {
        "developer": dev,
        "overlap": overlap,
        "summary_text": sobha_context.format_overlap_for_llm(dev or None),
    }


@app.get("/admin/sobha-project")
async def get_sobha_project_endpoint(
    q: str = "",
    authorization: str | None = Header(default=None),
) -> dict:
    """Look up Sobha projects by name, community, or partial match (Friday tool).

    Searches the sobha_projects table for any of:
      - exact project_name match (case-insensitive)
      - substring in project_name
      - substring in community
      - status filter (when q is "completed" / "ongoing")

    Returns up to 8 matches with full KPI data. Empty `matches` array
    when nothing matches — Friday is told to say so honestly rather
    than fabricate.
    """
    _require_admin(authorization)
    query = (q or "").strip()
    if not query:
        return {"query": "", "matches": [], "note": "empty query"}

    try:
        # Pull all projects (49 rows; cheap enough to filter in Python
        # rather than craft per-field ilike queries through PostgREST).
        rows = (
            sb.client()
            .table("sobha_projects")
            .select("*")
            .limit(100)
            .execute()
            .data
        ) or []
    except Exception as e:
        log.warning("sobha-project lookup failed: %s", e)
        return {"query": query, "matches": [], "error": str(e)[:200]}

    q_l = query.lower()

    def _score(r: dict) -> int:
        name_l = (r.get("project_name") or "").lower()
        comm_l = (r.get("community") or "").lower()
        status_l = (r.get("status") or "").lower()
        # Higher is better; 0 means no match.
        if name_l == q_l:
            return 100
        if q_l in name_l:
            return 80
        if q_l in comm_l:
            return 50
        if q_l == status_l or q_l in ("completed", "ongoing") and q_l == status_l:
            return 30
        return 0

    scored = [(r, _score(r)) for r in rows]
    matches = [r for r, s in sorted(scored, key=lambda t: t[1], reverse=True) if s > 0][:8]

    return {
        "query": query,
        "match_count": len(matches),
        "matches": [
            {
                "project_name":      m.get("project_name"),
                "community":         m.get("community"),
                "status":            m.get("status"),
                "launch_date":       m.get("launch_date"),
                "total_units":       m.get("total_units"),
                "saleable_area_sqft": m.get("saleable_area_sqft"),
                "sold_pct":          m.get("sold_pct"),
                "avg_sold_psf_aed":  m.get("avg_sold_psf_aed"),
                "construction_pct":  m.get("construction_pct"),
                "project_topline_aed": m.get("project_topline_aed"),
            }
            for m in matches
        ],
    }


# ─── Friday text chat endpoint ─────────────────────────────────────
# Mirrors the voice agent's persona for the dashboard's text chat
# surface. Same Sobha grounding (portfolio summary in the system
# message), same scope (RE + proptech + finance + revenue ideas),
# same "I don't have clarity" fallback discipline.
#
# v1: no tool calls — the LLM answers from its system prompt + the
# user message + (when relevant) the latest brief/CEO-scan blob the
# frontend sends as additional context. Tool-calling parity with the
# voice agent's MCP loop is a v2 effort.
#
# Streams via SSE so the dashboard renders responses incrementally.

_FRIDAY_TEXT_PERSONA = """\
You are Friday — text-chat sibling of the voice assistant Francis Alfred
uses on the MDI dashboard. Same identity, same posture, same scope.

# ⚠ DATA FIDELITY — THE ONE RULE THAT OVERRIDES EVERYTHING

Every concrete fact in your reply MUST come from a context block
appearing later in this system prompt (Sobha portfolio · Live market
backdrop · Latest morning brief · Today's agenda · Today's major news
· Memory · Chip-fired fact pack). You may NOT invent: meeting titles,
attendee names, project sales percentages, deal sizes, capital
amounts, dates, quotes, project names, or any other specific
attribute. If a context block is missing or empty, say so plainly
("No meetings on the calendar today, boss." / "Today's brief hasn't
been generated yet — the pipeline runs at 05:00 GST.").

Generic market knowledge (e.g. "S&P 500 is a US stock index") is fine.
Specifics about Sobha, its peers, or today's events are NOT.

If you violate this rule the boss loses trust in the tool. He has
explicitly flagged hallucinated meetings (e.g. "Sobha Central Phase
III investor discussions") and fabricated sales figures (e.g.
"97% of units sold") as the failure mode you must avoid.

# Identity
- Real-estate analyst + proptech scout + finance reader. Not a generic AI.
- Sobha is a Dubai luxury developer expanding into Abu Dhabi, USA (Texas),
  Australia (Brisbane). You sit on top of MDI: 50 scouts, daily brief,
  per-region CEO scan, on-demand deep dives.
- CEO altitude: market structure, capital flows, segment shifts, revenue
  opportunities. Tactical detail only when asked.

# Address
- Call him "boss" by default, "Mr Alfred" when formal.

# Scope
1. Real estate (primary)
2. Proptech / construction tech / AI tools competitors are adopting
3. Capital + finance (sovereign wealth, sukuk, REIT, rates, FX, mortgage)
4. Revenue ideas (branded residences, JVs, alternative asset classes)
5. Macro shifts that change Sobha's playing field

# When the boss asks for "the morning brief" (or "today's read" /
  "what's the brief" / "summarize the brief")

REQUIRED format:
  1. The narrative — recite from the "Latest morning brief · Narrative"
     section verbatim or near-verbatim. 2-3 sentences max.
  2. By category — list each non-empty category from the brief's
     "By category" section as a bullet. Format: "**<category>:** <bullets>".
     Skip categories that don't appear in the block.
  3. Where to act today — copy the "Where to act" pointers verbatim
     as bullets. Skip if the block has no pointers.
  4. Today's meetings — copy lines from the "Today's agenda" block
     verbatim. If the block is empty: "No meetings on the calendar today."

ANTI-PATTERNS (do not do):
  ✘ Add framing sentences not in the brief ("The market is showing...")
  ✘ Invent meeting topics or attendees
  ✘ Quote specific Sobha sales figures that don't appear in the brief
  ✘ Predict outcomes
  ✘ Use the phrase "Evidence suggests..." or "This may benefit Sobha..."

# When the boss asks "what's on today" / "any news" / "any meetings"

Combine the "Today's agenda" block and "Today's major news" block.
Bullet list. If either is empty, say so.

# When the boss asks anything else (real estate / market / capital / Sobha)

1. Lead with the answer (1-3 sentences max), drawing facts from the
   context blocks.
2. Cite the source by name: "Per the brief," / "The market snapshot
   shows," / "The Sobha portfolio summary lists,"
3. Optional implication for Sobha — name a specific Sobha project
   ONLY if it appears in the portfolio summary. Don't fabricate.
4. Action only when asked.

# When you don't have data — REQUIRED phrasing
  "I don't have clarity on that, boss. Could you tell me [specific Q]?
   Feed me [specific data] and I'll work it out."

# Voice
British English. Dry wit fine. No sycophancy. No "Of course!", "Great
question!", "Let me check...". Just answer.

Markdown is fine but keep it minimal — no large headers, no nested
lists. Bold for emphasis sparingly.
"""


# ─── Friday: contextual suggestion chips ───────────────────────────
# GET /api/friday-suggestions returns up to 5 suggestions only when
# the underlying data is robust enough to support a sharp answer.
# Each chip carries a `context_id` that /api/friday-chat uses to
# inject the matching fact-pack into the LLM prompt.

def _safe_isoformat_window(days: int) -> str:
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    return (_dt.now(_tz.utc) - _td(days=days)).isoformat()


@app.get("/api/friday-suggestions")
async def friday_suggestions(limit: int = 5) -> dict:
    """5 dynamic chips, each backed by robust data.

    Order of priority: brief → recent investigated events → patterns
    → topic clusters → Sobha-exposure events → markets fallback.
    """
    suggestions: list[dict] = []
    sb_client = sb.client()

    # 1. Today's brief (always show if it exists)
    try:
        from datetime import datetime as _dt, timezone as _tz
        today = _dt.now(_tz.utc).date().isoformat()
        latest = (
            sb_client.table("daily_briefs")
            .select("day,signal_count")
            .order("day", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if latest and (latest[0].get("signal_count") or 0) > 0:
            day = latest[0].get("day") or today
            suggestions.append({
                "kind": "brief",
                "label": "What's the morning brief?",
                "prompt": "Give me today's morning brief — lead with the market backdrop, then the top 2-3 moves.",
                "context_id": f"brief:{day}",
                "icon": "📰",
            })
    except Exception as e:
        log.info("[friday-sugg] brief check failed: %s", e)

    # 2. Recent investigated events (max 2 chips; need full facts + tldr)
    try:
        seven_d = _safe_isoformat_window(7)
        events = (
            sb_client.table("market_events")
            .select("id,project_name,developer_slug,headline,event_type,event_date")
            .gte("event_date", seven_d)
            .order("event_date", desc=True)
            .limit(20)
            .execute()
            .data
            or []
        )
        # Need an event_implications row with non-empty tldr to qualify
        event_ids_seen = 0
        for e in events:
            if event_ids_seen >= 2:
                break
            eid = e.get("id")
            if not eid:
                continue
            try:
                impl_rows = (
                    sb_client.table("event_implications")
                    .select("tldr")
                    .eq("event_id", eid)
                    .limit(1)
                    .execute()
                    .data
                    or []
                )
                tldr = (impl_rows[0].get("tldr") if impl_rows else "") or ""
                if not tldr or "failed" in tldr.lower():
                    continue
            except Exception:
                continue

            project = (e.get("project_name") or "").strip()
            dev = (e.get("developer_slug") or "").strip()
            headline = (e.get("headline") or "").strip()
            if project and dev:
                label = f"{dev.title()}'s {project} — what's the read?"
                prompt = f"Walk me through {dev.title()}'s {project} — fact pack, scenarios, what we should watch."
            elif project:
                label = f"{project} — what's the read?"
                prompt = f"Walk me through {project} — fact pack, scenarios, what we should watch."
            elif headline:
                label = headline[:60]
                prompt = f"Walk me through this event: {headline}"
            else:
                continue

            suggestions.append({
                "kind": "investigated_event",
                "label": label,
                "prompt": prompt,
                "context_id": f"event:{eid}",
                "icon": "🏗",
            })
            event_ids_seen += 1
    except Exception as e:
        log.info("[friday-sugg] events check failed: %s", e)

    # 3. Patterns with high evidence count
    try:
        seven_d_date = _safe_isoformat_window(7)[:10]
        patterns = (
            sb_client.table("market_patterns")
            .select("key,title,claim,region,evidence_count,sobha_implication")
            .gte("week_start", seven_d_date)
            .order("evidence_count", desc=True)
            .limit(10)
            .execute()
            .data
            or []
        )
        for p in patterns[:1]:  # take strongest pattern only
            ec = p.get("evidence_count") or 0
            if ec < 3:
                continue
            title = (p.get("title") or "").strip()
            if not title:
                continue
            suggestions.append({
                "kind": "pattern",
                "label": title[:60],
                "prompt": f"Tell me about this pattern: {title}. What's driving it and what should we watch?",
                "context_id": f"pattern:{p.get('key')}",
                "icon": "📊",
            })
    except Exception as e:
        log.info("[friday-sugg] patterns check failed: %s", e)

    # 4. Topic cluster (decision_tag with ≥3 signals last 7d)
    try:
        seven_d = _safe_isoformat_window(7)
        rows = (
            sb_client.table("signals")
            .select("decision_tag,region")
            .gte("last_seen_at", seven_d)
            .eq("archived", False)
            .limit(500)
            .execute()
            .data
            or []
        )
        from collections import Counter
        tag_region: Counter = Counter()
        for r in rows:
            tag = (r.get("decision_tag") or "").strip().lower()
            region = (r.get("region") or "").strip().lower()
            if tag and tag != "none":
                tag_region[(tag, region or "global")] += 1
        if tag_region:
            (top_tag, top_region), top_count = tag_region.most_common(1)[0]
            if top_count >= 3:
                region_label = top_region if top_region != "global" else "across regions"
                # Friendly tag → label
                tag_label = {
                    "launch": "What launched this week",
                    "pricing": "Pricing moves this week",
                    "capital": "Capital flows this week",
                    "risk": "Risk signals this week",
                }.get(top_tag, f"{top_tag.title()} signals this week")
                suggestions.append({
                    "kind": "topic_cluster",
                    "label": f"{tag_label} ({region_label})",
                    "prompt": f"What {top_tag} signals have we surfaced in {region_label} this week?",
                    "context_id": f"cluster:{top_tag}:{top_region}",
                    "icon": "⚡",
                })
    except Exception as e:
        log.info("[friday-sugg] cluster check failed: %s", e)

    # 5. Markets fallback (always available; deduped if we already have 5)
    if len(suggestions) < limit:
        suggestions.append({
            "kind": "macro",
            "label": "Quick market check",
            "prompt": "Quick market snapshot — Brent, S&P, gold, AED, USD/INR. One paragraph.",
            "context_id": "market:snapshot",
            "icon": "💱",
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(suggestions[:limit]),
        "suggestions": suggestions[:limit],
    }


# ─── context_id resolver ────────────────────────────────────────────
async def _resolve_context_block(context_id: str) -> str:
    """Turn a chip's context_id into a fact-pack the LLM can read.

    Shapes:
      brief:<YYYY-MM-DD>      → today's brief narrative + top signals
      event:<id>              → investigated event facts + scenarios
      pattern:<key>           → market_patterns row
      cluster:<tag>:<region>  → top signals matching the cluster
      market:snapshot         → live market quote line
    """
    if not context_id:
        return ""
    try:
        kind, _, rest = context_id.partition(":")
        if kind == "brief":
            row = (
                sb.client().table("daily_briefs").select("*")
                .eq("day", rest).limit(1).execute().data or [None]
            )[0]
            if row:
                tops = row.get("top_signals") or []
                top_lines = [
                    f"  - (#{t.get('signal_id')}) {t.get('headline')}"
                    for t in tops[:5] if t.get("headline")
                ]
                return (
                    f"# Today's brief ({rest})\n"
                    f"{row.get('narrative','')[:2000]}\n\n"
                    f"# Top signals\n" + "\n".join(top_lines)
                )

        if kind == "event":
            try:
                eid = int(rest)
            except ValueError:
                return ""
            ev = (
                sb.client().table("market_events").select("*")
                .eq("id", eid).limit(1).execute().data or [None]
            )[0]
            if not ev:
                return ""
            facts_row = (
                sb.client().table("event_facts").select("*")
                .eq("event_id", eid).limit(1).execute().data or [None]
            )[0] or {}
            impl_row = (
                sb.client().table("event_implications").select("*")
                .eq("event_id", eid).limit(1).execute().data or [None]
            )[0] or {}
            scenarios = impl_row.get("scenarios") or []
            sc_lines = [
                f"  - [{s.get('stance','?')}] {s.get('claim','')}"
                for s in scenarios[:3]
            ]
            return (
                f"# Investigated event #{eid}\n"
                f"  Project: {ev.get('project_name','?')}\n"
                f"  Developer: {ev.get('developer_slug','?')}\n"
                f"  Type: {ev.get('event_type','?')}  Date: {ev.get('event_date','?')}\n"
                f"# Facts\n  {facts_row.get('facts') or {}}\n"
                f"# TLDR\n  {impl_row.get('tldr','—')}\n"
                f"# Scenarios\n" + "\n".join(sc_lines)
            )

        if kind == "pattern":
            row = (
                sb.client().table("market_patterns").select("*")
                .eq("key", rest).order("week_start", desc=True)
                .limit(1).execute().data or [None]
            )[0]
            if row:
                return (
                    f"# Pattern: {row.get('title','?')}\n"
                    f"  Stance: {row.get('stance','?')}\n"
                    f"  Region: {row.get('region','?')}\n"
                    f"  Evidence count: {row.get('evidence_count','?')}\n"
                    f"  Claim: {row.get('claim','?')}\n"
                    f"  Sobha implication: {row.get('sobha_implication','—')}\n"
                    f"  Action: {row.get('action','—')}"
                )

        if kind == "cluster":
            tag, _, region = rest.partition(":")
            seven_d = _safe_isoformat_window(7)
            q = (
                sb.client().table("signals").select("id,headline,dek,priority,dev_slug,last_seen_at,region")
                .eq("decision_tag", tag).gte("last_seen_at", seven_d)
                .eq("archived", False)
                .order("last_seen_at", desc=True)
                .limit(8)
            )
            if region and region != "global":
                q = q.eq("region", region)
            rows = q.execute().data or []
            if rows:
                lines = [
                    f"  - (#{r.get('id')}) [{r.get('priority','?')}/{r.get('dev_slug','?')}/{r.get('region','?')}] "
                    f"{r.get('headline','')}: {r.get('dek','')[:140]}"
                    for r in rows
                ]
                return f"# Cluster: {tag} ({region})\n" + "\n".join(lines)

        if kind == "market":
            from tools import market_snapshot as _ms
            snap = await _ms.get_snapshot()
            return f"# Market snapshot\n  {snap.get('one_liner') or 'unavailable'}"
    except Exception as e:
        log.info("[friday-ctx] resolve failed for %s: %s", context_id, e)
    return ""


# ─── RAG memory + persistence ─────────────────────────────────────
async def _retrieve_memory(query_text: str, k: int = 4) -> list[dict]:
    """Retrieve top-K similar past Friday turns via pgvector match_friday_memory RPC."""
    if not query_text.strip():
        return []
    try:
        from tools.nvidia_llm import embed
        vec = await embed(query_text)
        if not vec:
            return []
        return (
            sb.client().rpc(
                "match_friday_memory",
                {"query_embedding": vec, "match_count": k, "similarity_threshold": 0.75},
            ).execute().data or []
        )
    except Exception as e:
        log.info("[friday-mem] retrieval failed: %s", e)
        return []


async def _persist_turn(
    session_id: str,
    role: str,
    content: str,
    context_id: str | None = None,
) -> None:
    """Persist one turn (user or assistant) with an embedding for future RAG."""
    if not content or not content.strip():
        return
    try:
        from tools.nvidia_llm import embed
        vec = await embed(content[:1500])
    except Exception:
        vec = None
    try:
        sb.client().table("friday_conversations").insert({
            "session_id": session_id,
            "role": role,
            "content": content[:4000],
            "context_id": context_id,
            "embedding": vec,
        }).execute()
    except Exception as e:
        log.info("[friday-mem] persist failed: %s", e)


# ─── Friday context helpers (added 2026-05-07) ─────────────────────
# Stakeholder directive: when the MD asks Friday anything, include
# today's meetings (from the SharePoint calendar) and today's major
# news headlines (from the signals table) in context. The voice agent
# inherits the same context_block via the system prompt.

# Dashboard URL — env override for staging; defaults to production.
import os as _os_friday
_DASHBOARD_BASE = (_os_friday.environ.get("DASHBOARD_BASE_URL")
                   or "https://sobha-mdi.vercel.app").rstrip("/")
_AGENDA_TTL_SEC = 5 * 60
_AGENDA_CACHE: dict[str, Any] = {"at": 0.0, "block": ""}


async def _today_agenda_block() -> str:
    """Compact text block of today's meetings from the MD's calendar.

    Hits the dashboard's /api/calendar SharePoint proxy. 5-min cache so
    every Friday turn doesn't re-pull the Excel. Empty string on any
    failure (fail-soft so a calendar outage never blocks a chat reply).
    """
    import time as _time
    now = _time.time()
    if _AGENDA_CACHE["block"] and (now - _AGENDA_CACHE["at"]) < _AGENDA_TTL_SEC:
        return _AGENDA_CACHE["block"]
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"{_DASHBOARD_BASE}/api/calendar")
            if resp.status_code != 200:
                return ""
            data = resp.json()
    except Exception as e:
        log.info("[friday-agenda] fetch failed: %s", type(e).__name__)
        return ""
    meetings = data.get("meetings") or []
    if not meetings:
        return ""
    # Filter to today (UTC; SharePoint Excel is timezone-naive but the
    # MD operates in GST so we widen the window slightly to catch both).
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    today = _dt.now(_tz.utc).date()
    todays: list[dict[str, Any]] = []
    for m in meetings:
        dtstr = m.get("datetime") or ""
        try:
            mt = _dt.fromisoformat(dtstr.replace("Z", "+00:00"))
            mday = mt.date()
            if abs((mday - today).days) <= 0:
                todays.append(m)
        except Exception:
            continue
    if not todays:
        # Show next 24h instead so an early-morning ask still gets coverage.
        for m in meetings:
            dtstr = m.get("datetime") or ""
            try:
                mt = _dt.fromisoformat(dtstr.replace("Z", "+00:00"))
                if 0 <= (mt - _dt.now(_tz.utc)).total_seconds() <= 86400:
                    todays.append(m)
            except Exception:
                continue
    if not todays:
        return ""
    bits = []
    for m in todays[:8]:
        dtstr = m.get("datetime") or ""
        try:
            mt = _dt.fromisoformat(dtstr.replace("Z", "+00:00"))
            time_str = mt.strftime("%H:%M GST")
        except Exception:
            time_str = "—"
        title = (m.get("title") or "Meeting").strip()
        loc = (m.get("location") or "").strip()
        attendees = (m.get("attendees") or "").strip()
        bits.append(
            f"  - {time_str} · {title}"
            + (f" · {loc}" if loc else "")
            + (f" · with {attendees}" if attendees else "")
        )
    block = "\n".join(bits)
    _AGENDA_CACHE["at"] = now
    _AGENDA_CACHE["block"] = block
    return block


async def _today_headlines_block(limit: int = 10) -> str:
    """Top priority signals from last 24h — neutral facts, no framing.

    Reads `signals` directly. No LLM call. Returns ~10 lines max so the
    Friday system prompt stays under the model's context window.
    """
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cutoff = (_dt.now(_tz.utc) - _td(hours=24)).isoformat()
        # 2026-05-11 fix: signals table has `urls text[]` (not `url`)
        # and has NO `source` column (we derive source from urls[0]
        # hostname client-side). Migrations 002 / 005 / 008 confirm
        # the actual column set. The wrong SELECT was making this
        # helper silently return [] — Friday saw an empty headlines
        # block and hallucinated content to fill the gap.
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,region,source_published_at,urls,first_seen_at")
            .gte("last_seen_at", cutoff)
            .eq("archived", False)
            .order("priority", desc=True)
            .order("source_published_at", desc=True)
            .limit(40)
            .execute()
            .data
            or []
        )
    except Exception as e:
        log.info("[friday-headlines] query failed: %s", e)
        return ""
    if not rows:
        return ""
    # Prefer high+medium priority; deduplicate by headline-stem to avoid
    # repeating the same story across regions.
    seen_stems: set[str] = set()
    picked: list[dict[str, Any]] = []
    for r in rows:
        if (r.get("priority") or "").lower() not in ("high", "medium"):
            continue
        stem = (r.get("headline") or "").lower().strip()[:60]
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        picked.append(r)
        if len(picked) >= limit:
            break
    if not picked:
        return ""
    bits = []
    for r in picked:
        region = r.get("region") or "global"
        headline = (r.get("headline") or "").strip()
        # signals table has no `source` column — derive a host hint
        # from urls[0] for the source attribution.
        urls = r.get("urls") or []
        src = "src"
        if isinstance(urls, list) and urls:
            try:
                from urllib.parse import urlparse as _up
                src = _up(urls[0]).hostname or "src"
                if src.startswith("www."):
                    src = src[4:]
            except Exception:
                pass
        bits.append(f"  - [{region}] {headline} ({src})")
    return "\n".join(bits)


async def _today_brief_block() -> str:
    """Latest daily brief content — narrative + per-category bullets.

    Added 2026-05-11 per stakeholder directive: "When asked to FRIDAY,
    it should give the brief or summary that he was already giving AND
    the meetings of the day from today's agenda." The agenda block is
    already injected via `_today_agenda_block`; this adds the actual
    brief content so Friday can recite the morning read on demand.

    Reads the latest row from `daily_briefs`. Returns a compact text
    block with narrative + each non-empty synthesis category. Empty
    string on any failure (fail-soft).
    """
    try:
        # 2026-05-11 fix: daily_briefs has NO standalone `tldr` column.
        # The TL;DR lives inside `pointers` JSONB (migration 003).
        # Asking for `tldr` made Supabase reject the SELECT → empty
        # rows → empty brief block → Friday hallucinated. Schema is:
        # id, day, narrative, top_signals, synthesis, signal_count,
        # generated_at, generator_model, pointers.
        rows = (
            sb.client().table("daily_briefs")
            .select("day,narrative,synthesis,top_signals,pointers,signal_count,generated_at")
            .order("day", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as e:
        log.info("[friday-brief] query failed: %s", e)
        return ""
    if not rows:
        return ""
    b = rows[0]
    day = b.get("day") or ""
    parts: list[str] = []
    parts.append(f"# Latest morning brief — {day}")
    narrative = (b.get("narrative") or "").strip()
    if narrative:
        # Keep the prose to ~1200 chars so the system prompt stays bounded.
        parts.append("Narrative:")
        parts.append(narrative[:1200])
    # TL;DR lives in pointers.tldr (not a standalone column)
    pointers_obj = b.get("pointers") or {}
    if isinstance(pointers_obj, dict):
        tldr = (pointers_obj.get("tldr") or "").strip()
        if tldr:
            parts.append(f"TLDR: {tldr[:400]}")
    synth = b.get("synthesis") or {}
    if isinstance(synth, dict) and synth:
        parts.append("By category:")
        for cat, body in synth.items():
            if not body:
                continue
            text_line = ""
            if isinstance(body, dict):
                bullets = body.get("bullets") or []
                if bullets and isinstance(bullets, list):
                    bullet_texts = []
                    for bb in bullets[:4]:
                        if isinstance(bb, dict):
                            t = (bb.get("text") or "").strip()
                            if t:
                                bullet_texts.append(t)
                        elif isinstance(bb, str):
                            bullet_texts.append(bb.strip())
                    if bullet_texts:
                        text_line = " · ".join(bullet_texts)[:400]
                if not text_line:
                    text_line = (body.get("synthesis") or "").strip()[:400]
            elif isinstance(body, str):
                text_line = body.strip()[:400]
            if text_line:
                parts.append(f"  - {cat}: {text_line}")
    # pointers is jsonb with shape {tldr, actions[]}; actions is the
    # bulleted "Where to act today" list. Sometimes the field has been
    # written as a flat list of strings in older briefs; handle both.
    pointers = b.get("pointers")
    actions = []
    if isinstance(pointers, dict):
        actions = pointers.get("actions") or []
    elif isinstance(pointers, list):
        actions = pointers
    if isinstance(actions, list) and actions:
        parts.append("Where to act:")
        for p in actions[:4]:
            if isinstance(p, dict):
                line = (p.get("text") or p.get("pointer") or "").strip()
                if line:
                    parts.append(f"  - {line[:200]}")
            elif isinstance(p, str):
                parts.append(f"  - {p.strip()[:200]}")
    block = "\n".join(parts)
    return block[:3500]  # hard ceiling so context stays bounded


@app.post("/api/friday-chat")
async def friday_chat(request: Request) -> StreamingResponse:
    """Text chat sibling of the Friday voice agent.

    Body:
      {
        "message": "user's text",
        "history": [{"role":"user|assistant","content":"..."}, ...],
        "context": "optional extra context blob",
        "context_id": "optional chip-fired context id (e.g. 'event:42')",
        "session_id": "client-supplied uuid for grouping turns"
      }

    Streams Server-Sent Events.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")

    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="missing message")
    history = body.get("history") or []
    extra_ctx = (body.get("context") or "").strip()
    context_id = (body.get("context_id") or "").strip()
    session_id = (body.get("session_id") or "").strip() or "anon"

    # Sobha portfolio context — same digest the voice agent gets.
    sobha_summary = sobha_context.format_for_llm("summary")

    # Resolve chip-fired context_id into a fact-pack
    chip_context = ""
    if context_id:
        chip_context = await _resolve_context_block(context_id)

    # Live market snapshot — same source the brief writer reads
    market_block = ""
    try:
        from tools import market_snapshot as _ms
        snap = await _ms.get_snapshot()
        market_block = snap.get("one_liner") or ""
    except Exception:
        pass

    # Today's agenda — fetch the MD's calendar from the dashboard's
    # SharePoint proxy. Same source the Today's Agenda tab reads from.
    # (Added 2026-05-07: stakeholder ask "when asked to FRIDAY it should
    # give all major news plus today's meetings from todays agenda.")
    agenda_block = await _today_agenda_block()
    # Today's headlines — top priority signals from last 24h, all
    # regions, no editorial framing. Pulled from the same `signals`
    # table the dashboard renders.
    headlines_block = await _today_headlines_block(limit=10)
    # Latest morning brief — narrative + per-category bullets + pointers.
    # Lets Friday recite the morning read when the MD asks for "the
    # brief" / "today's read" / "what's the morning brief" (2026-05-11).
    brief_block = await _today_brief_block()

    # RAG memory: retrieve relevant past Friday turns
    memory_rows = await _retrieve_memory(user_msg, k=4)
    memory_block = ""
    if memory_rows:
        bits = []
        for m in memory_rows:
            sim = m.get("similarity") or 0
            bits.append(f"  [{sim:.2f}] ({m.get('role','?')}): {m.get('content','')[:280]}")
        memory_block = "# Memory — relevant past Friday turns\n" + "\n".join(bits)

    system_block = (
        f"{_FRIDAY_TEXT_PERSONA}\n\n"
        f"# Sobha portfolio context (current)\n{sobha_summary}\n"
    )
    if market_block:
        system_block += f"\n# Live market backdrop\n  {market_block}\n"
    if brief_block:
        system_block += f"\n{brief_block}\n"
    if agenda_block:
        system_block += f"\n# Today's agenda — MD's meetings\n{agenda_block}\n"
    if headlines_block:
        system_block += f"\n# Today's major news (top signals, last 24h)\n{headlines_block}\n"
    if memory_block:
        system_block += f"\n{memory_block}\n"
    if chip_context:
        system_block += f"\n# Chip-fired fact pack (context_id={context_id})\n{chip_context}\n"
    if extra_ctx:
        system_block += f"\n# Additional context provided this turn\n{extra_ctx}\n"

    messages = [{"role": "system", "content": system_block}]
    for h in history[-10:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:4000]})
    messages.append({"role": "user", "content": user_msg[:4000]})

    # Persist the user turn now (don't block on embedding)
    asyncio.create_task(_persist_turn(session_id, "user", user_msg, context_id or None))

    from settings import settings as _settings
    if not (_settings.llm_base_url and _settings.llm_api_key and _settings.llm_model):
        raise HTTPException(status_code=503, detail="LLM not configured")

    import httpx
    body_to_llm = {
        "model": _settings.llm_model,
        "messages": messages,
        "temperature": 0.3,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {_settings.llm_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async def _stream():
        url = f"{_settings.llm_base_url.rstrip('/')}/chat/completions"
        assembled: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=body_to_llm, headers=headers) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        yield f'data: {{"error":"upstream {resp.status_code}: {err_body.decode()[:200]}"}}\n\n'
                        return
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            import json as _json
                            chunk = _json.loads(payload)
                            delta = (
                                ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                                or ""
                            )
                            if delta:
                                assembled.append(delta)
                                import json as _json2
                                yield f"data: {_json2.dumps({'delta': delta})}\n\n"
                        except Exception:
                            continue
            yield 'data: {"done": true}\n\n'
        except Exception as e:
            log.exception("friday-chat stream failed")
            yield f'data: {{"error":"{str(e)[:200]}"}}\n\n'
        finally:
            # Persist assistant turn after stream closes (non-blocking)
            full = "".join(assembled).strip()
            if full:
                asyncio.create_task(_persist_turn(session_id, "assistant", full, context_id or None))

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Friday TTS — sentence-stream proxy to Deepgram ────────────────
@app.post("/api/friday-tts")
async def friday_tts(request: Request) -> StreamingResponse:
    """Proxy text → Deepgram TTS audio. Returns audio/mpeg streaming.

    Body: {"text": "...", "voice": "aura-2-thalia-en"}

    Frontend buffers chat deltas, splits on sentence boundaries, and
    fires this endpoint per-sentence to feed an audio queue. Result:
    voice starts within ~1.5s of first sentence and continues smoothly.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="missing text")
    voice = (body.get("voice") or "aura-2-thalia-en").strip()

    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not deepgram_key:
        raise HTTPException(status_code=503, detail="DEEPGRAM_API_KEY not configured")

    import httpx
    dg_url = f"https://api.deepgram.com/v1/speak?model={voice}&encoding=mp3"
    dg_headers = {
        "Authorization": f"Token {deepgram_key}",
        "Content-Type": "application/json",
    }
    dg_body = {"text": text[:1500]}

    async def _audio_stream():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", dg_url, json=dg_body, headers=dg_headers) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        log.warning("Deepgram TTS %s: %s", resp.status_code, err[:200])
                        return
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except Exception as e:
            log.exception("friday-tts stream failed: %s", e)

    return StreamingResponse(
        _audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Conversation rating (for future fine-tune corpus) ─────────────
@app.post("/api/friday-rate")
async def friday_rate(request: Request) -> dict:
    """Rate a past Friday turn (thumbs up=1, down=-1).

    Body: {"turn_id": <bigint>, "rating": 1 | -1}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    turn_id = body.get("turn_id")
    rating = body.get("rating")
    if not isinstance(turn_id, int) or rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="bad params")
    try:
        sb.client().table("friday_conversations").update({"rating": rating}).eq("id", turn_id).execute()
    except Exception as e:
        log.exception("friday-rate failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ok": True, "turn_id": turn_id, "rating": rating}


@app.post("/admin/sync-sobha-mis")
async def sync_sobha_mis(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict:
    """Upload a fresh Sobha MIS xlsx → upsert sobha_projects.

    Auth: optional Bearer secret if ADMIN_SECRET env is set. When
    unset (dev), endpoint accepts any caller.
    """
    expected = os.environ.get("ADMIN_SECRET", "").strip()
    if expected:
        if not authorization or authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid auth")

    fname = (file.filename or "").lower()
    if not fname.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="expected .xlsx upload")

    # Save to a temp file (openpyxl needs a real path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        count = sobha_context.sync_from_xlsx(tmp_path)
    except Exception as e:
        log.exception("sobha MIS sync failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return {"ok": True, "projects_synced": count, "summary": sobha_context.get_portfolio_summary()}


@app.post("/run/daily-brief")
async def run_daily_brief() -> dict:
    try:
        return await daily_brief_flow.run()
    except Exception as e:
        log.exception("manual daily brief failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/run/ceo-scan")
async def run_ceo_scan(cadence: str = "daily") -> dict:
    """Manually trigger the CEO scan for all regions. ?cadence=daily|weekly."""
    if cadence not in ("daily", "weekly"):
        raise HTTPException(status_code=400, detail="cadence must be 'daily' or 'weekly'")
    try:
        return await ceo_scan_flow.run(cadence=cadence)
    except Exception as e:
        log.exception("manual ceo scan failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/patterns")
async def get_patterns(region: str | None = None, days: int = 7) -> dict:
    """Run cross-signal pattern detection across recent signals.

    Returns deterministic patterns (no LLM) — dev activity surges,
    topic clusters, micro-market clusters, tech-keyword clusters.

    Params:
      region — optional filter ('dubai' / 'abu_dhabi' / 'usa' / 'australia' / 'other').
               When set, only patterns whose region matches (or is None) are returned.
      days   — recency window for the recent set (default 7). Baseline is fixed at 30.
    """
    from datetime import datetime, timedelta, timezone
    from tools import pattern_detection

    now = datetime.now(timezone.utc)
    since_recent = (now - timedelta(days=days)).isoformat()
    since_baseline = (now - timedelta(days=30)).isoformat()

    sb_client = sb.client()
    base_query = (
        sb_client.table("signals")
        .select("id,headline,dek,category,decision_tag,priority,confidence,"
                "dev_slug,region,country_code,source_published_at,last_seen_at")
        .eq("archived", False)
    )
    recent = (
        base_query.gte("last_seen_at", since_recent)
        .limit(300).execute().data or []
    )
    baseline = (
        sb_client.table("signals")
        .select("id,dev_slug,region,last_seen_at")
        .eq("archived", False)
        .gte("last_seen_at", since_baseline)
        .lt("last_seen_at", since_recent)
        .limit(1000).execute().data or []
    )

    patterns = pattern_detection.detect_all(recent, baseline)
    if region:
        patterns = [p for p in patterns if not p.get("region") or p["region"] == region]
    # LLM-prose layer — wraps each pattern's deterministic claim with
    # a Sobha-specific implication + action. Cached per (key, day) so
    # dashboard polls don't spend tokens repeatedly.
    try:
        patterns = await pattern_detection.enrich_with_prose(patterns)
    except Exception as e:
        log.warning("pattern prose enrichment failed: %s — returning deterministic only", e)
    # Persist to market_patterns (idempotent on (key, week_start)) so
    # the historical view in Phase 5.3 has data to chart. Defensive:
    # if migration 010 hasn't applied, this just logs and returns.
    try:
        pattern_detection.persist_patterns(patterns)
    except Exception as e:
        log.warning("persist_patterns wrapper failed: %s", e)
    return {
        "patterns": patterns,
        "meta": {
            "recent_count": len(recent),
            "baseline_count": len(baseline),
            "window_days": days,
            "filtered_region": region,
        },
    }


@app.get("/patterns/history")
async def get_pattern_history(
    weeks: int = 8,
    region: str | None = None,
    key: str | None = None,
) -> dict:
    """Historical view of persisted patterns. Drives the Phase 5.3
    'patterns over time' frontend route once that ships.

    Params:
      weeks  — how many weeks back to fetch (default 8)
      region — optional region filter
      key    — optional pattern-key filter (e.g. 'dev_surge:emaar:7d')
    """
    rows = pattern_detection.fetch_pattern_history(
        weeks=weeks, region=region, pattern_key=key,
    )
    return {"history": rows, "meta": {"weeks": weeks, "region": region, "key": key}}


@app.get("/run/ceo-scan/stream")
async def run_ceo_scan_stream(cadence: str = "daily") -> StreamingResponse:
    """SSE-streamed CEO scan. Same flow as /run/ceo-scan but emits a
    text/event-stream with progress events per phase, so the dashboard
    can show actual backend state instead of faux progress.

    Conscious GET-with-side-effects: EventSource only supports GET.
    Frontend uses EventSource to read events; the flow runs once
    per connection.

    Event sequence:
      start, fetching_signals, fetched_signals, bucketed,
      fetching_events, fetched_events,
      scanning_region (×5), scanned_region (×5),
      done | error
    """
    if cadence not in ("daily", "weekly"):
        raise HTTPException(status_code=400, detail="cadence must be 'daily' or 'weekly'")

    async def event_stream():
        try:
            async for ev in ceo_scan_flow.run_streaming(cadence=cadence):
                yield f"data: {_json.dumps(ev)}\n\n"
        except Exception as e:
            log.exception("streaming ceo scan failed")
            yield f"data: {_json.dumps({'event': 'error', 'detail': str(e)[:200]})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable proxies / browsers buffering the stream so events
            # arrive promptly rather than in one chunk at the end.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/run/pipeline")
async def run_pipeline(force_linkedin: int = 0) -> dict:
    """Manually trigger the full chained pipeline:
       ingest → daily_brief → ceo_scan (daily).

    Same sequence the scheduler runs at 05:00 + 13:00 GST. Use this when
    you want a fresh top-to-bottom refresh on demand. Processing only
    fires when ingest succeeds — matches the scheduler's dependency
    rule ('don't process before signals are in').

    Query params:
      ?force_linkedin=1  — bypass the linkedin_scout's UTC morning-only
                            gate (lets you trigger a LinkedIn pull at
                            any time of day; usually used right after a
                            new leader is added to the Leader Feed).
    """
    out: dict = {"ingest": None, "brief": None, "ceo_scan_daily": None}
    try:
        out["ingest"] = await ingest_flow.run(force_linkedin=bool(force_linkedin))
    except Exception as e:
        log.exception("manual pipeline: ingest failed")
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}") from e

    try:
        out["brief"] = await daily_brief_flow.run()
    except Exception as e:
        log.warning("manual pipeline: brief failed but ingest ok: %s", e)
        out["brief_error"] = str(e)

    try:
        out["ceo_scan_daily"] = await ceo_scan_flow.run(cadence="daily")
    except Exception as e:
        log.warning("manual pipeline: ceo_scan failed but ingest ok: %s", e)
        out["ceo_scan_error"] = str(e)

    return out


@app.get("/ceo-scan")
async def get_ceo_scan(
    region: str | None = None,
    cadence: str = "daily",
    day: str | None = None,
) -> dict:
    """Latest CEO scan output. Returns one region or the full set.

    - `region` is one of dubai|abu_dhabi|usa|australia|other (omit for all)
    - `cadence` is daily|weekly (default daily)
    - `day` YYYY-MM-DD (omit for latest)
    """
    if cadence not in ("daily", "weekly"):
        raise HTTPException(status_code=400, detail="cadence must be 'daily' or 'weekly'")
    q = sb.client().table("ceo_scans").select("*").eq("cadence", cadence)
    if region:
        if region not in ceo_scan_flow.REGIONS:
            raise HTTPException(
                status_code=400,
                detail=f"region must be one of: {list(ceo_scan_flow.REGIONS)}",
            )
        q = q.eq("region", region)
    if day:
        q = q.eq("scan_day", day)
    rows = q.order("scan_day", desc=True).limit(20).execute().data or []
    if not rows:
        return {"scans": []}
    # If no region filter, group most-recent-per-region.
    if not region:
        latest: dict[str, dict] = {}
        for r in rows:
            rg = r.get("region")
            if rg and rg not in latest:
                latest[rg] = r
        scans = list(latest.values())
    else:
        scans = rows[:1]
    # Phase F2: enrich every scan with signal_url_index covering every
    # signal_id referenced in section bullets, so per-section bullet
    # rendering can attach inline ↗ source chips.
    try:
        ids: set[int] = set()
        for scan in scans:
            output = scan.get("output") or {}
            for sec in (output.get("sections") or []):
                for b in (sec.get("bullets") or []):
                    if isinstance(b, dict):
                        for sid in b.get("signal_ids") or []:
                            try:
                                ids.add(int(sid))
                            except Exception:
                                pass
        idx = _build_signal_url_index(list(ids))
        for scan in scans:
            scan["signal_url_index"] = idx
    except Exception as e:
        log.info("ceo-scan: url-index failed: %s", e)
    return {"scans": scans}


@app.post("/deep-dive/{dev_slug}")
async def deep_dive(dev_slug: str) -> dict:
    dev_slug = dev_slug.lower().strip()
    if dev_slug not in VALID_DEVS:
        raise HTTPException(
            status_code=400,
            detail=f"dev_slug must be one of: {sorted(VALID_DEVS)}",
        )
    try:
        return await deep_dive_crew.run(dev_slug)
    except Exception as e:
        log.exception("deep_dive failed for %s", dev_slug)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/daily-brief")
async def get_daily_brief(day: str | None = None) -> dict:
    """Latest brief, or a specific day with ?day=YYYY-MM-DD.

    Phase F2: enriches the brief with `signal_url_index` covering every
    signal_id referenced in top_signals + per-bucket synthesis bullets +
    narrative (#N) markers, so the dashboard renders inline ↗ source chips.
    """
    q = sb.client().table("daily_briefs").select("*")
    q = q.eq("day", day) if day else q.order("day", desc=True).limit(1)
    data = q.execute().data or []
    if not data:
        raise HTTPException(status_code=404, detail="No brief found.")
    brief = data[0]
    try:
        ids: set[int] = set()
        # Top-signal cards each have a signal_id field.
        for t in brief.get("top_signals") or []:
            sid = t.get("signal_id") or t.get("id")
            if sid is not None:
                try:
                    ids.add(int(sid))
                except Exception:
                    pass
        # Per-bucket synthesis bullets carry signal_ids (Phase F1).
        synth = brief.get("synthesis") or {}
        if isinstance(synth, dict):
            for bucket in synth.values():
                if not isinstance(bucket, dict):
                    continue
                for sid in bucket.get("signal_ids") or []:
                    try:
                        ids.add(int(sid))
                    except Exception:
                        pass
                for b in bucket.get("bullets") or []:
                    if isinstance(b, dict):
                        for sid in b.get("signal_ids") or []:
                            try:
                                ids.add(int(sid))
                            except Exception:
                                pass
        # Narrative (#N) markers — kept by writer/check tasks, render as ↗ chips.
        # 2026-04-30 (audit-round 2): only strip the ALWAYS-bad cite
        # tokens at this layer:
        #   1. (#) — empty cite token from LLM that forgot the id
        #   2. (#FOO), (#MARKETDATA) — non-numeric placeholders
        # Numeric cites like (#42) are LEFT IN regardless of url_index
        # status — the frontend renders them as plain `#N` text when
        # the URL doesn't resolve, preserving the reference instead of
        # silently swallowing it. Audit caught the over-correction
        # where every chip was being stripped.
        narrative = brief.get("narrative") or ""
        import re as _re
        if narrative:
            narrative = _re.sub(r"\s?\(#\)", "", str(narrative))
            narrative = _re.sub(r"\s?\(#[A-Z][A-Z0-9_-]*\)", "", narrative)
            # Tidy up double-spaces / orphan punctuation left by drops.
            narrative = _re.sub(r"\s{2,}", " ", narrative)
            narrative = _re.sub(r"\s+([.,;:])", r"\1", narrative)
            brief["narrative"] = narrative
        # Collect numeric cite IDs from the cleaned narrative so the
        # url_index covers them.
        for m in _re.findall(r"\(#(\d+)\)", str(narrative)):
            try:
                ids.add(int(m))
            except Exception:
                pass
        brief["signal_url_index"] = _build_signal_url_index(list(ids))
    except Exception as e:
        log.info("daily-brief: url-index failed: %s", e)
        brief["signal_url_index"] = {}
    return brief


@app.get("/debug/azure-intel")
async def debug_azure_intel() -> dict:
    """Probe the Azure AD SP + MD Intelligence app wiring.

    Reports which env vars are set, attempts a token, then pings a
    list of common endpoints to discover what the app exposes. Remove
    once integration is live.
    """
    return await azure_intel.diagnose()


@app.get("/api/signals/by-dev")
async def signals_by_dev(slug: str, days: int = 90, limit: int = 30) -> dict:
    """All recent signals for a tracked developer slug.

    2026-04-30 round 8: powers Friday's `query_signals_by_dev_slug` tool
    so the MD can ask "show me everything Aldar did this quarter."

    Args:
        slug:  dev_slug (emaar / damac / aldar / nakheel / sobha / etc.)
        days:  lookback window (default 90 = ~Q1)
        limit: max rows (default 30, hard cap 100)
    """
    if not slug:
        return {"signals": [], "error": "slug required"}
    days = max(1, min(int(days), 365))
    limit = max(1, min(int(limit), 100))
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,subcategory,"
                    "decision_tag,region,country_code,last_seen_at,urls,"
                    "corroboration_count")
            .eq("dev_slug", slug.lower())
            .eq("archived", False)
            .gte("last_seen_at", since)
            .order("last_seen_at", desc=True).limit(limit).execute().data or []
        )
    except Exception as e:
        log.info("signals/by-dev: fetch failed (%s) — retrying without subcategory", e)
        # Pre-migration-019 fallback.
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,decision_tag,"
                    "region,country_code,last_seen_at,urls,corroboration_count")
            .eq("dev_slug", slug.lower())
            .eq("archived", False)
            .gte("last_seen_at", since)
            .order("last_seen_at", desc=True).limit(limit).execute().data or []
        )
    return {
        "slug": slug.lower(),
        "days": days,
        "count": len(rows),
        "signals": rows,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/signals/by-subcategory")
async def signals_by_subcategory(subcategory: str, days: int = 14, limit: int = 30) -> dict:
    """Signals filtered by LLM-assigned subcategory (round 7 + later).

    Powers Friday's `query_subcategory` tool. Examples the MD might ask:
      "What frontier AI happened this week?" → subcategory=frontier_ai
      "Any new ConTech innovation?"           → subcategory=contech
      "Show visa policy changes."             → subcategory=regulatory_visa
      "Recent PropTech moves."                → subcategory=proptech

    Returns rows with subcategory != NULL only — legacy signals
    (subcategory=NULL) aren't reachable here. They show via the keyword
    fallback on themed pages.
    """
    if not subcategory:
        return {"signals": [], "error": "subcategory required"}
    days = max(1, min(int(days), 90))
    limit = max(1, min(int(limit), 100))
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("signals")
            .select("id,headline,dek,priority,category,subcategory,"
                    "dev_slug,region,country_code,last_seen_at,urls,"
                    "corroboration_count")
            .eq("subcategory", subcategory.lower())
            .eq("archived", False)
            .gte("last_seen_at", since)
            .order("last_seen_at", desc=True).limit(limit).execute().data or []
        )
    except Exception as e:
        # Migration 019 not run yet.
        log.info("signals/by-subcategory: subcategory column missing — %s", e)
        return {"signals": [], "subcategory": subcategory.lower(),
                "data_gap": "subcategory column not yet migrated"}
    return {
        "subcategory": subcategory.lower(),
        "days": days,
        "count": len(rows),
        "signals": rows,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/metrics/history")
async def metrics_history(metric_key: str, days: int = 90) -> dict:
    """Time-series for a metric_key from metrics_daily.

    Powers Friday's `query_metrics_history` tool. Examples:
      metric_key="eibor_3m"           → EIBOR 3M last quarter
      metric_key="hpi_uk_london"      → London HPI history
      metric_key="yf_BZ_F"            → Brent Yahoo cache
      metric_key="yf_GC_F"            → Gold history
      metric_key="hpi_singapore"      → Singapore HPI

    Returns chronological points + first/last values + computed pct
    move so Friday can verbalise "EIBOR 3M moved from 4.45 to 4.50,
    up 1.1% over the quarter" without doing math itself.
    """
    if not metric_key:
        return {"points": [], "error": "metric_key required"}
    days = max(1, min(int(days), 365))
    from datetime import timedelta as _td
    since = (datetime.now(timezone.utc) - _td(days=days)).isoformat()
    try:
        rows = (
            sb.client().table("metrics_daily")
            .select("captured_at,value_num,raw_json,source")
            .eq("metric_key", metric_key)
            .gte("captured_at", since)
            .order("captured_at").execute().data or []
        )
    except Exception as e:
        log.info("metrics/history: fetch failed: %s", e)
        return {"points": [], "metric_key": metric_key, "data_gap": "fetch failed"}
    if not rows:
        return {"points": [], "metric_key": metric_key, "days": days,
                "data_gap": "no points in window"}
    first = float(rows[0].get("value_num") or 0)
    last  = float(rows[-1].get("value_num") or 0)
    pct = None
    if first and last:
        pct = round((last - first) / first * 100.0, 2)
    return {
        "metric_key": metric_key,
        "days": days,
        "n_points": len(rows),
        "first_value": first,
        "last_value":  last,
        "pct_change":  pct,
        "first_at": rows[0].get("captured_at"),
        "last_at":  rows[-1].get("captured_at"),
        "points": [
            {"captured_at": r.get("captured_at"),
             "value": r.get("value_num"),
             "source": r.get("source")}
            for r in rows
        ],
    }


@app.get("/signals/recent")
async def signals_recent(limit: int = 10) -> dict:
    """Quick list of recent signals for picking an id to investigate."""
    rows = (
        sb.client()
        .table("signals")
        .select("id,priority,confidence,dev_slug,headline,dek,last_seen_at")
        .eq("archived", False)
        .order("last_seen_at", desc=True)
        .limit(max(1, min(limit, 50)))
        .execute()
        .data
        or []
    )
    return {"signals": rows}


@app.post("/signals/{signal_id}/feedback")
async def signal_feedback(signal_id: int, verdict: str) -> dict:
    """MD feedback: 'useful' or 'noise'.

    - useful: no-op other than recording (already kept).
    - noise : archive the signal so it stops showing on the dashboard.
    Both verdicts are persisted to signal_feedback so we can use them
    later to re-tune classifier weights.
    """
    v = (verdict or "").strip().lower()
    if v not in ("useful", "noise"):
        raise HTTPException(status_code=400, detail="verdict must be 'useful' or 'noise'")
    client = sb.client()
    try:
        client.table("signal_feedback").upsert(
            {"user_id": "md", "signal_id": signal_id, "verdict": v},
            on_conflict="user_id,signal_id",
        ).execute()
    except Exception as e:
        log.warning("feedback persist failed for signal=%s: %s", signal_id, e)
    if v == "noise":
        try:
            client.table("signals").update({"archived": True}).eq("id", signal_id).execute()
        except Exception as e:
            log.warning("archive failed for signal=%s: %s", signal_id, e)
    return {"ok": True, "signal_id": signal_id, "verdict": v}


# ─── Investigation layer ──────────────────────────────────────────────────
@app.post("/investigate/{signal_id}")
async def investigate(signal_id: int) -> dict:
    """Manually trigger an investigation on any signal (any priority).

    High-priority signals auto-investigate on ingest; this endpoint is
    for medium/low signals the MD wants to dig deeper on, or for
    re-running a past investigation with fresher evidence.
    """
    # Trigger label: if an event already exists for this signal we treat
    # this as a re-investigation; otherwise it's a manual first-fire.
    try:
        existing = (
            sb.client()
            .table("market_events")
            .select("id")
            .eq("seed_signal_id", signal_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        trigger = "re-investigate" if existing else "manual"
    except Exception:
        trigger = "manual"
    try:
        return await investigation_flow.run(signal_id, trigger=trigger)
    except Exception as e:
        log.exception("investigate failed for signal_id=%s", signal_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/events/{event_id}/runs")
async def event_runs(event_id: int, limit: int = 10) -> dict:
    """Investigation run history — preserves how truth evolved."""
    rows = (
        sb.client()
        .table("event_runs")
        .select("*")
        .eq("event_id", event_id)
        .order("run_at", desc=True)
        .limit(max(1, min(limit, 50)))
        .execute()
        .data
        or []
    )
    return {"runs": rows}


@app.get("/projects")
async def list_projects(
    limit: int = 50,
    developer_slug: str | None = None,
    branded_only: bool = False,
) -> dict:
    """Recent projects — canonical layer that rolls up market_events."""
    q = (
        sb.client()
        .table("projects")
        .select("*")
        .order("last_updated_at", desc=True)
        .limit(max(1, min(limit, 200)))
    )
    if developer_slug:
        q = q.eq("developer_slug", developer_slug.lower().strip())
    if branded_only:
        q = q.eq("branded", True)
    return {"projects": q.execute().data or []}


@app.get("/projects/{project_id}")
async def get_project(project_id: int) -> dict:
    """One project + its rolled-up events."""
    proj = (
        sb.client()
        .table("projects")
        .select("*")
        .eq("id", project_id)
        .limit(1)
        .execute()
        .data
    )
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    events = (
        sb.client()
        .table("market_events")
        .select("id,slug,event_type,event_date,status,headline,last_updated_at")
        .eq("project_id", project_id)
        .order("last_updated_at", desc=True)
        .execute()
        .data
        or []
    )
    return {"project": proj[0], "events": events}


@app.get("/events")
async def list_events(
    limit: int = 20,
    event_type: str | None = None,
    developer_slug: str | None = None,
) -> dict:
    """Recent market_events with joined facts + implications for the dashboard.

    2026-04-30 audit-round: switched to event_implications(*) so the
    listing endpoint includes scenarios — was missing previously while
    the single-event endpoint at /events/{id} had it. Means the
    dashboard's listing render now matches the drawer exactly.
    """
    q = (
        sb.client()
        .table("market_events")
        .select("*, event_facts(*), event_implications(*)")
        .order("last_updated_at", desc=True)
        .limit(max(1, min(limit, 100)))
    )
    if event_type:
        q = q.eq("event_type", event_type)
    if developer_slug:
        q = q.eq("developer_slug", developer_slug)
    return {"events": q.execute().data or []}


@app.get("/events/{event_id}")
async def get_event(event_id: int) -> dict:
    """Full evidence pack for one event — for the dashboard drill-down."""
    ev = (
        sb.client()
        .table("market_events")
        .select("*, event_facts(*), event_implications(*), event_evidence(*)")
        .eq("id", event_id)
        .limit(1)
        .execute()
        .data
    )
    if not ev:
        raise HTTPException(status_code=404, detail="event not found")
    return ev[0]
