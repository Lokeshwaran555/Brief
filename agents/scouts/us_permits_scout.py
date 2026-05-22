"""US city building-permits scout — Austin Socrata API.

Building permits are construction-start signals. A permit issued today
means foundation work is days-to-weeks out — material for absorption-
velocity calls and competitor-launch forecasting in Texas multifamily
markets.

Direct GET against city-run Socrata — free, no key, no rate limit
beyond polite use.

Source:
  - Austin: dataset 3syk-w9eu (Issued Construction Permits) — rich
    schema with permit_class_mapped (Commercial/Residential),
    work_class, applieddate, issue_date, address, lat/long, link to
    permit detail page. Refreshed daily.

Filtering strategy: pull last 30 days of *Commercial* / *Multifamily*
/ *Apartment* permits only, sorted newest first. Single-family +
electrical-only permits are too noisy.

Dallas / Houston / NSW / ABS deliberately deferred (verified 2026-04-26):
  - Dallas: the open-data portal lists 5+ "Building Permits" datasets
    dated today but all 404 on SODA — they're metadata-only entries,
    not API-exposed. The only SODA-exposed dataset (e7gq-4sah) is
    frozen at 2019.
  - Houston: open-data only exposes aggregated monthly XLS, not per-
    permit feed — would need XLS parsing for limited signal density.
  - NSW Planning Portal API now requires registered API key, which
    breaks our free-no-key scout pattern.
  - ABS Building Approvals is monthly national/state XLSX —
    one-signal-per-month aggregate; defer until xlsx parsing is in
    the project's dependency footprint.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)


# ─── Austin ─────────────────────────────────────────────────────
# Socrata SoQL: pull commercial / multifamily / new-construction
# permits issued in the last 30 days. We exclude single-family and
# pure trade permits (electrical-only / plumbing-only) at the API
# level so we don't pay for the full ~10k/month firehose.
AUSTIN_DATASET = "3syk-w9eu"
AUSTIN_SOQL_FRAGMENT = (
    "permit_class_mapped='Commercial' AND "
    "issue_date > '{since}' AND "
    "(upper(work_class) like '%NEW%' OR "
    " upper(work_class) like '%ADDITION%' OR "
    " upper(permit_class) like '%MULTI%' OR "
    " upper(permit_class) like '%APARTMENT%')"
)


DAYS_BACK = 30
LIMIT_PER_CITY = 50


def _dedup_key(city: str, permit_no: Any, address: str) -> str:
    head = f"us-permits:{city}:{permit_no}:{(address or '').strip().lower()[:80]}"
    return hashlib.sha256(head.encode()).hexdigest()


def _format_value_usd(v: Any) -> str:
    try:
        v = float(v or 0)
    except Exception:
        return ""
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v > 0:
        return f"${v:,.0f}"
    return ""


def _austin_row(r: dict[str, Any]) -> dict[str, Any] | None:
    permit_no = r.get("permit_number")
    if not permit_no:
        return None
    work_class = (r.get("work_class") or "").strip()
    permit_class = (r.get("permit_class") or "").strip()
    desc = (r.get("description") or "").strip()
    address = (r.get("original_address1") or "").strip()
    issued = (r.get("issue_date") or "")[:10]
    council = (r.get("council_district") or "").strip()
    detail_url = (r.get("link") or {}).get("url") if isinstance(r.get("link"), dict) else None

    headline_bits = [f"Austin permit issued [{permit_class or 'Commercial'}]"]
    if address:
        headline_bits.append(f"at {address}")
    if work_class:
        headline_bits.append(f"({work_class})")
    headline = " ".join(headline_bits)[:280]

    summary_bits = []
    if desc:
        summary_bits.append(desc)
    if issued:
        summary_bits.append(f"Issued: {issued}")
    if council:
        summary_bits.append(f"District: {council}")
    summary = " · ".join(summary_bits)[:600]

    return {
        "source": "us_permits:austin",
        "source_url": detail_url or "https://data.austintexas.gov/d/3syk-w9eu",
        "title": headline,
        "summary": summary,
        "raw_json": {
            "city": "Austin",
            "permit_number": permit_no,
            "permit_class": permit_class,
            "permit_class_mapped": r.get("permit_class_mapped"),
            "work_class": work_class,
            "permittype": r.get("permittype"),
            "description": desc,
            "address": address,
            "zip": r.get("original_zip"),
            "council_district": council,
            "applied_date": r.get("applieddate"),
            "issue_date": r.get("issue_date"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "published_date": issued,
            "region": "usa",
            "country_code": "US",
            "msa_hint": "austin",
            "category_hint": "permit",
        },
        "dedup_key": _dedup_key("austin", permit_no, address),
    }


async def _fetch_socrata(
    client: httpx.AsyncClient,
    *,
    city: str,
    host: str,
    dataset: str,
    where: str,
    order: str,
    mapper,
) -> list[dict[str, Any]]:
    url = (
        f"https://{host}/resource/{dataset}.json"
        f"?$where={quote(where)}"
        f"&$order={quote(order)}"
        f"&$limit={LIMIT_PER_CITY}"
    )
    try:
        resp = await client.get(
            url,
            headers={"User-Agent": "Sobha MDI US-permits scout (contact: ops@sobha.com)"},
            timeout=15.0,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        log.warning("[us_permits:%s] fetch failed: %s", city, e)
        return []
    out: list[dict[str, Any]] = []
    for r in rows or []:
        m = mapper(r)
        if m:
            out.append(m)
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
    today = datetime.now(timezone.utc).date()
    since = (today - timedelta(days=DAYS_BACK)).isoformat()

    austin_where = AUSTIN_SOQL_FRAGMENT.format(since=since)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        austin = await _fetch_socrata(
            client,
            city="austin",
            host="data.austintexas.gov",
            dataset=AUSTIN_DATASET,
            where=austin_where,
            order="issue_date DESC",
            mapper=_austin_row,
        )

    deduped = _dedupe(austin)
    log.info(
        "[us_permits] austin=%d deduped=%d",
        len(austin), len(deduped),
    )
    return deduped[:limit]
