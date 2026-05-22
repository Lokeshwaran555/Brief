"""Australia building-approvals scout — ABS Time Series Workbooks.

ABS publishes monthly dwelling-approvals figures by state in the
8731.0 Building Approvals release. Headline tables expose
month-by-month total dwelling units approved (private + total
sectors). MoM movement is the most-watched leading indicator of
Australian residential pipeline pressure.

We pull the two states most relevant to Sobha's AU footprint:
  - Table 01 (NSW)  → Sydney coverage
  - Table 03 (QLD)  → Brisbane priority

Each run discovers the *current* release month from the
'latest-release' page (URL pattern shifts every month, e.g.
feb-2026/8731001.xlsx → mar-2026/8731001.xlsx) and downloads the
two table workbooks. Emits one signal per state per release with
MoM change vs the prior month.

Free, no key. Fail-soft on any fetch / parse error — returns [].
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import openpyxl

log = logging.getLogger(__name__)


ABS_LATEST = "https://www.abs.gov.au/statistics/industry/building-and-construction/building-approvals-australia/latest-release"

# Table 01 = NSW, Table 03 = QLD. The 'Total (Type of Building) ;
# Total Sectors' column is the headline figure (private + public).
TABLES: list[dict[str, str]] = [
    {"file": "8731001.xlsx", "state": "New South Wales", "region": "australia", "city_hint": "sydney"},
    {"file": "8731003.xlsx", "state": "Queensland",      "region": "australia", "city_hint": "brisbane"},
]

# Column index inside the Data1 sheet for the headline "Total dwelling
# units; <state>; Total (Type of Building); Total Sectors" series.
# Verified for 8731001 + 8731003 against the Feb 2026 release —
# col 6 in zero-indexed terms (column G in the workbook).
TOTAL_SECTORS_COL = 6


def _dedup_key(state: str, period_iso: str) -> str:
    head = f"abs_building:{state}:{period_iso}"
    return hashlib.sha256(head.encode()).hexdigest()


async def _discover_release_dir(client: httpx.AsyncClient) -> str | None:
    """Scrape the latest-release page for the current month's path
    fragment (e.g. 'feb-2026'). Returns None if the page or pattern
    can't be found — caller logs and skips.
    """
    try:
        resp = await client.get(
            ABS_LATEST,
            headers={"User-Agent": "Sobha MDI AU-building scout (contact: ops@sobha.com)"},
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[au_building] latest-release page fetch failed: %s", e)
        return None

    m = re.search(
        r"building-approvals-australia/([a-z]+-\d{4})/8731001\.xlsx",
        resp.text,
    )
    return m.group(1) if m else None


async def _fetch_table(
    client: httpx.AsyncClient, release_dir: str, table: dict[str, str],
) -> dict[str, Any] | None:
    url = (
        f"https://www.abs.gov.au/statistics/industry/building-and-construction/"
        f"building-approvals-australia/{release_dir}/{table['file']}"
    )
    try:
        resp = await client.get(
            url,
            headers={"User-Agent": "Sobha MDI AU-building scout (contact: ops@sobha.com)"},
            timeout=20.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.warning("[au_building:%s] xlsx fetch failed: %s", table["state"], e)
        return None

    try:
        wb = openpyxl.load_workbook(io.BytesIO(resp.content), data_only=True, read_only=True)
        ws = wb["Data1"]
        rows = list(ws.iter_rows(values_only=True))
    except Exception as e:
        log.warning("[au_building:%s] xlsx parse failed: %s", table["state"], e)
        return None

    # Data rows start where col0 is a datetime. Last 2 such rows are
    # the latest + prior month.
    data_rows = [r for r in rows if isinstance(r[0], datetime)]
    if len(data_rows) < 2:
        log.warning("[au_building:%s] insufficient data rows: %d", table["state"], len(data_rows))
        return None

    latest_period = data_rows[-1][0]
    prior_period = data_rows[-2][0]
    latest_total = data_rows[-1][TOTAL_SECTORS_COL]
    prior_total = data_rows[-2][TOTAL_SECTORS_COL]

    if not isinstance(latest_total, (int, float)) or not isinstance(prior_total, (int, float)):
        log.warning("[au_building:%s] non-numeric headline cells", table["state"])
        return None

    mom_pct = (
        ((latest_total - prior_total) / prior_total) * 100.0
        if prior_total else None
    )

    return {
        "state": table["state"],
        "region": table["region"],
        "city_hint": table["city_hint"],
        "release_dir": release_dir,
        "latest_period": latest_period,
        "prior_period": prior_period,
        "latest_total": latest_total,
        "prior_total": prior_total,
        "mom_pct": mom_pct,
        "url": url,
    }


def _row_to_intel(d: dict[str, Any]) -> dict[str, Any]:
    state = d["state"]
    period = d["latest_period"].strftime("%b %Y")
    period_iso = d["latest_period"].strftime("%Y-%m-01")
    total = int(d["latest_total"])
    prior = int(d["prior_total"])
    pct = d["mom_pct"]

    direction = "↑" if (pct or 0) >= 0 else "↓"
    pct_str = f"{abs(pct):.1f}%" if pct is not None else "—"

    headline = (
        f"ABS: {state} dwelling approvals {period} = {total:,} "
        f"({direction}{pct_str} MoM, prior {prior:,})"
    )
    summary = (
        f"ABS 8731.0 Building Approvals release for {d['release_dir']}. "
        f"Total dwelling units approved (Total Sectors): "
        f"{period}={total:,} vs prior month {prior:,}. "
        f"MoM change {direction}{pct_str}."
    )

    return {
        "source": f"abs_building:{d['city_hint']}",
        "source_url": d["url"],
        "title": headline[:300],
        "summary": summary[:600],
        "raw_json": {
            "state": state,
            "city_hint": d["city_hint"],
            "release_dir": d["release_dir"],
            "period": period_iso,
            "prior_period": d["prior_period"].strftime("%Y-%m-01"),
            "latest_total": total,
            "prior_total": prior,
            "mom_pct": pct,
            "published_date": period_iso,
            "region": d["region"],
            "country_code": "AU",
            "category_hint": "macro_indicator",
        },
        "dedup_key": _dedup_key(state, period_iso),
    }


async def run(limit: int = 5) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        release_dir = await _discover_release_dir(client)
        if not release_dir:
            return []
        results = await asyncio.gather(
            *[_fetch_table(client, release_dir, t) for t in TABLES],
            return_exceptions=False,
        )

    out = [_row_to_intel(r) for r in results if r]
    log.info(
        "[au_building] release=%s tables=%d emitted=%d",
        release_dir, len(TABLES), len(out),
    )
    return out[:limit]
