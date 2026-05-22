"""Dubai Land Department project-registry scout — direct gateway API.

Pulls structured project-registration data straight from the DLD's
open-data gateway (the same endpoint that powers
dubailand.gov.ae/en/open-data/real-estate-data). Every off-plan
project must be registered here with project number, developer,
unit count, value, escrow account, and status — *before* marketing
begins. Entries appear weeks before press releases.

API endpoint reverse-engineered 2026-04-26 from the public page:
  POST https://gateway.dubailand.gov.ae/open-data/projects
  Header  ConsumerId: <public consumer ID baked into the page>
  Body    P_FROM_DATE / P_TO_DATE in MM/DD/YYYY,
          P_DATE_TYPE = "1" (Start) / "3" (Adoption) / "4" (Completion),
          P_PRJ_STATUS, P_TAKE/P_SKIP as strings, etc.

Three independent pulls per run, ranked by signal type:
  1. Recently-registered (P_DATE_TYPE=1, last 60 days) — early-launch
     leak. Catches "Soulever by Beyond"-style new projects months
     before any press coverage.
  2. Recently-adopted (P_DATE_TYPE=3, last 60 days) — final RERA
     approval; project moves from paperwork to launchable.
  3. Distress signals (P_PRJ_STATUS in CANCELLED /
     UNDER_CANCELATION_*, last 180 days) — project stalls, capital
     pressure, developer trouble.

No auth. No paid API. ConsumerId is the same public string the
public page uses. Captcha layer (featureGate) was off during
verification; if it ever returns we'll fall back to fail-soft empty.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)


# Public consumer ID baked into dubailand.gov.ae open-data page.
# Same value used by the human-facing UI — verified live 2026-04-26.
DLD_CONSUMER_ID = "gkb3WvEG0rY9eilwXC0P2pTz8UzvLj9F"
DLD_PROJECTS_URL = "https://gateway.dubailand.gov.ae/open-data/projects"


# Developer-name → dev_slug. Same canonical slugs as ingest_flow.
# DLD developer names are upper-cased and contain LLC suffix —
# normalize before substring-matching.
DEV_MATCHERS: list[tuple[str, list[str]]] = [
    ("emaar",     ["emaar"]),
    ("damac",     ["damac"]),
    ("aldar",     ["aldar"]),
    ("nakheel",   ["nakheel"]),
    ("modon",     ["modon"]),
    ("dubaih",    ["dubai holding", "meraas", "dubai properties"]),
    ("binghatti", ["binghatti"]),
    ("azizi",     ["azizi"]),
    ("ellington", ["ellington"]),
    ("omniyat",   ["omniyat"]),
    ("deyaar",    ["deyaar"]),
    ("sobha",     ["sobha"]),
    ("unionprops", ["union properties"]),
]


def _dev_slug(developer_en: str) -> str | None:
    t = (developer_en or "").lower()
    for slug, needles in DEV_MATCHERS:
        if any(n in t for n in needles):
            return slug
    return None


def _dedup_key(project_number: Any, status: str, source_tag: str) -> str:
    """Stable per (project, status, pull-tag).

    Including status means a project moving Active → Cancelled produces
    a NEW signal rather than upserting silently. Including source_tag
    lets the same project appear once for registration + once for
    adoption (different lifecycle moments).
    """
    head = f"dld:{project_number}:{status}:{source_tag}"
    return hashlib.sha256(head.encode()).hexdigest()


def _format_value_aed(v: Any) -> str:
    try:
        v = float(v or 0)
    except Exception:
        return ""
    if v >= 1_000_000_000:
        return f"AED {v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"AED {v/1_000_000:.0f}M"
    if v > 0:
        return f"AED {v:,.0f}"
    return ""


def _build_headline(row: dict[str, Any], source_tag: str) -> str:
    """Synthesize a human-readable headline from a structured row.

    The classifier expects article-shaped input — give it one. Lead
    with the verb that matches the source_tag so priority routing works.
    """
    project = (row.get("PROJECT_EN") or "").strip()
    dev = (row.get("DEVELOPER_EN") or "").strip()
    status = (row.get("PROJECT_STATUS") or "").strip()
    units = row.get("CNT_TOTAL") or 0
    value = _format_value_aed(row.get("PROJECT_VALUE"))
    area = (row.get("AREA_EN") or "").strip()

    parts = []
    if source_tag == "registration":
        parts.append(f"DLD registered: {project}")
    elif source_tag == "adoption":
        parts.append(f"DLD adopted (RERA approved): {project}")
    elif source_tag == "distress":
        parts.append(f"DLD status update [{status}]: {project}")
    else:
        parts.append(f"DLD: {project}")

    if dev:
        parts.append(f"by {dev.strip()}")
    detail_bits = []
    if units:
        detail_bits.append(f"{units} units")
    if value:
        detail_bits.append(value)
    if area:
        detail_bits.append(area)
    if detail_bits:
        parts.append("(" + ", ".join(detail_bits) + ")")
    return " ".join(parts)[:300]


def _build_summary(row: dict[str, Any]) -> str:
    desc = (row.get("DESCRIPTION_EN") or "").strip()
    escrow = (row.get("ESCROW_ACCOUNT_NUMBER") or "").strip()
    start = (row.get("START_DATE") or "")[:10]
    end = (row.get("END_DATE") or "")[:10]
    completion = (row.get("COMPLETION_DATE") or "")[:10]
    pct = row.get("PERCENT_COMPLETED")
    master = (row.get("MASTER_PROJECT_EN") or "").strip()

    bits = []
    if desc:
        bits.append(desc)
    if start:
        bits.append(f"Start: {start}")
    if end:
        bits.append(f"End: {end}")
    if completion:
        bits.append(f"Completion: {completion}")
    if pct not in (None, ""):
        bits.append(f"Progress: {pct}%")
    if escrow:
        bits.append(f"Escrow: {escrow}")
    if master:
        bits.append(f"Master: {master}")
    return " · ".join(bits)[:600]


def _published_at(row: dict[str, Any], source_tag: str) -> str | None:
    """Pick the lifecycle date that triggered this pull. The classifier
    uses this to compute freshness decay."""
    if source_tag == "registration":
        v = row.get("START_DATE") or row.get("ADOPTION_DATE") or row.get("INSPECTION_DATE")
    elif source_tag == "adoption":
        v = row.get("ADOPTION_DATE") or row.get("INSPECTION_DATE")
    elif source_tag == "distress":
        # Distress events have no dedicated timestamp; INSPECTION_DATE
        # is the closest proxy for "when DLD last touched the record."
        v = row.get("INSPECTION_DATE") or row.get("ADOPTION_DATE")
    else:
        v = row.get("START_DATE")
    if not v:
        return None
    return str(v)[:19]  # ISO-ish prefix


def _row_to_intel(row: dict[str, Any], source_tag: str) -> dict[str, Any]:
    project_no = row.get("PROJECT_NUMBER")
    status = row.get("PROJECT_STATUS") or ""
    headline = _build_headline(row, source_tag)
    summary = _build_summary(row)
    dev = _dev_slug(row.get("DEVELOPER_EN") or "")

    # Source URL: open-data page itself (the project registry has no
    # per-project canonical URL exposed publicly). We add the project
    # number as a fragment so the URL is still distinct per signal.
    source_url = (
        f"https://dubailand.gov.ae/en/open-data/real-estate-data/"
        f"#project-{project_no}"
    )

    return {
        "source": f"dld_projects:{source_tag}",
        "source_url": source_url,
        "title": headline,
        "summary": summary,
        "raw_json": {
            "project_number": project_no,
            "project_en": row.get("PROJECT_EN"),
            "developer_en": row.get("DEVELOPER_EN"),
            "developer_number": row.get("DEVELOPER_NUMBER"),
            "project_status": status,
            "project_value": row.get("PROJECT_VALUE"),
            "escrow_account_number": row.get("ESCROW_ACCOUNT_NUMBER"),
            "start_date": row.get("START_DATE"),
            "end_date": row.get("END_DATE"),
            "adoption_date": row.get("ADOPTION_DATE"),
            "completion_date": row.get("COMPLETION_DATE"),
            "inspection_date": row.get("INSPECTION_DATE"),
            "percent_completed": row.get("PERCENT_COMPLETED"),
            "area_en": row.get("AREA_EN"),
            "zone_en": row.get("ZONE_EN"),
            "cnt_total": row.get("CNT_TOTAL"),
            "cnt_unit": row.get("CNT_UNIT"),
            "cnt_villa": row.get("CNT_VILLA"),
            "cnt_building": row.get("CNT_BUILDING"),
            "master_project_en": row.get("MASTER_PROJECT_EN"),
            "prj_type_en": row.get("PRJ_TYPE_EN"),
            "published_date": _published_at(row, source_tag),
            "region": "dubai",
            "country_code": "AE",
            "dev_slug_hint": dev,
            "category_hint": "registry",
            "lifecycle_event": source_tag,
        },
        "dedup_key": _dedup_key(project_no, status, source_tag),
    }


async def _post(client: httpx.AsyncClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        resp = await client.post(
            DLD_PROJECTS_URL,
            json=body,
            headers={
                "ConsumerId": DLD_CONSUMER_ID,
                "Content-Type": "application/json",
                "Origin": "https://dubailand.gov.ae",
                "Referer": "https://dubailand.gov.ae/en/open-data/real-estate-data/",
                "User-Agent": "Sobha MDI DLD-projects scout (contact: ops@sobha.com)",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("[dld_projects] POST failed: %s", e)
        return []

    if data.get("responseCode") != 200:
        log.warning("[dld_projects] non-200 responseCode=%s errors=%s",
                    data.get("responseCode"), data.get("validationErrorsList"))
        return []

    response = data.get("response") or {}
    return response.get("result") or []


async def _pull_window(
    client: httpx.AsyncClient,
    *,
    days_back: int,
    date_type: str,
    status: str,
    source_tag: str,
    take: int = 50,
) -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    frm = today - timedelta(days=days_back)
    body = {
        "P_FROM_DATE": frm.strftime("%m/%d/%Y"),
        "P_TO_DATE":   today.strftime("%m/%d/%Y"),
        "P_DATE_TYPE": date_type,
        "P_PRJ_TYPE_ID": "",
        "P_PRJ_STATUS": status,
        "P_ZONE_ID": "",
        "P_AREA_ID": "",
        "P_TAKE": str(take),
        "P_SKIP": "0",
        "P_SORT": "PROJECT_START_DATE_DESC",
    }
    rows = await _post(client, body)
    return [_row_to_intel(r, source_tag) for r in rows if r.get("PROJECT_NUMBER")]


async def run(limit: int = 40) -> list[dict[str, Any]]:
    """Pull 3 lifecycle windows in parallel; dedupe; return up to `limit`.

    Three windows:
      registration  — new project registrations (last 60d, P_DATE_TYPE=1)
      adoption      — RERA-adopted (last 60d, P_DATE_TYPE=3)
      distress      — cancelled / under-cancellation in last 180d
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        registration_task = _pull_window(
            client, days_back=60, date_type="1", status="",
            source_tag="registration", take=50,
        )
        adoption_task = _pull_window(
            client, days_back=60, date_type="3", status="",
            source_tag="adoption", take=50,
        )
        # Distress: pull cancellation states one at a time — the API
        # only accepts a single status value per call.
        distress_statuses = (
            "CANCELLED",
            "UNDER_CANCELATION_DECISION",
            "UNDER_CANCELATION_NOTIFICATION",
        )
        distress_tasks = [
            _pull_window(
                client, days_back=180, date_type="1", status=s,
                source_tag="distress", take=20,
            )
            for s in distress_statuses
        ]

        results = await asyncio.gather(
            registration_task, adoption_task, *distress_tasks,
            return_exceptions=False,
        )

    flat: list[dict[str, Any]] = []
    for batch in results:
        flat.extend(batch)

    # Dedup on (project_number, status, source_tag) — already encoded
    # in dedup_key, but also collapse duplicates from overlapping pulls
    # (e.g. a brand-new project shows up in BOTH registration and
    # adoption pulls). We keep the registration variant since "newly
    # registered" is the higher-leverage signal.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    # Ordering matters here: registration first (we keep it), then
    # adoption, then distress. Within each, preserve API order.
    for it in flat:
        k = it["dedup_key"]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)

    log.info(
        "[dld_projects] reg=%d adop=%d distress=%d total=%d deduped=%d",
        len(results[0]), len(results[1]),
        sum(len(b) for b in results[2:]),
        len(flat), len(out),
    )
    return out[:limit]
