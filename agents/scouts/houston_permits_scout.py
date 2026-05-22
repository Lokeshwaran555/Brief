"""Houston residential building-permits scout — monthly summary XLSX.

Houston's open-data portal exposes only a monthly summary
spreadsheet, not a per-permit feed. The schema is simple — Year,
Month, Single Family, Multi-Family, Residential. Latest month at
file-tail.

Lower granularity than the Austin scout (one signal per month,
city-wide aggregate), but Texas-MSA permit volume is a real macro
indicator for Sobha's US footprint. MoM movement in Multi-Family
counts is the closest US analog to Brisbane's ABS approval series.

Verified 2026-04-26: data ends Dec 2024 (16 months stale at the time
of writing). Houston refreshes irregularly; we mark `stale_data=True`
in raw_json when the latest month is older than 90 days so the
classifier deprioritizes appropriately. Run still emits the row so
the freshness layer can decide.

Free, no key, no rate limit.
"""
from __future__ import annotations

import hashlib
import io
import logging
from datetime import date, datetime, timezone
from typing import Any

import httpx
import openpyxl

log = logging.getLogger(__name__)


HOUSTON_URL = (
    "https://data.houstontx.gov/dataset/"
    "525772a1-0b4b-4d55-9b43-4d3b589fe1b7/resource/"
    "c9cef716-4d81-4dac-9b05-d19acddf159f/download/"
    "summary-residential-permits.xlsx"
)

STALE_AFTER_DAYS = 90


def _dedup_key(period_iso: str) -> str:
    return hashlib.sha256(f"houston_permits:{period_iso}".encode()).hexdigest()


async def run(limit: int = 2) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                HOUSTON_URL,
                headers={"User-Agent": "Sobha MDI Houston-permits scout (contact: ops@sobha.com)"},
                timeout=20.0,
            )
            resp.raise_for_status()
    except Exception as e:
        log.warning("[houston_permits] xlsx fetch failed: %s", e)
        return []

    try:
        wb = openpyxl.load_workbook(io.BytesIO(resp.content), data_only=True, read_only=True)
        ws = wb["Permits"]
        rows = list(ws.iter_rows(values_only=True))
    except Exception as e:
        log.warning("[houston_permits] xlsx parse failed: %s", e)
        return []

    # Header row 0 = ('Year','Month','Single Family','Multi-Family','Residential')
    data_rows = [
        r for r in rows[1:]
        if isinstance(r[0], int) and isinstance(r[1], int)
    ]
    if len(data_rows) < 2:
        log.warning("[houston_permits] insufficient data rows: %d", len(data_rows))
        return []

    latest = data_rows[-1]
    prior = data_rows[-2]
    y, m, sf, mf, res = latest
    py, pm, psf, pmf, pres = prior

    period = date(y, m, 1)
    prior_period = date(py, pm, 1)
    age_days = (date.today() - period).days
    stale = age_days > STALE_AFTER_DAYS

    def _pct(now: int | None, before: int | None) -> float | None:
        if not isinstance(now, (int, float)) or not isinstance(before, (int, float)) or not before:
            return None
        return ((now - before) / before) * 100.0

    mf_pct = _pct(mf, pmf)
    res_pct = _pct(res, pres)

    direction = lambda v: "↑" if (v or 0) >= 0 else "↓"  # noqa: E731
    fmt_pct = lambda v: f"{abs(v):.1f}%" if v is not None else "—"  # noqa: E731

    period_label = period.strftime("%b %Y")
    headline = (
        f"Houston permits {period_label}: residential={res:,} "
        f"({direction(res_pct)}{fmt_pct(res_pct)} MoM), "
        f"multifamily={mf:,} ({direction(mf_pct)}{fmt_pct(mf_pct)} MoM)"
    )
    summary_bits = [
        f"Houston Public Works monthly summary, {period_label}.",
        f"Single Family={sf:,} (prior {psf:,}).",
        f"Multi-Family={mf:,} (prior {pmf:,}).",
        f"Total Residential={res:,} (prior {pres:,}).",
    ]
    if stale:
        summary_bits.append(
            f"NOTE: latest month is {age_days}d old — Houston open-data refresh has lagged."
        )
    summary = " ".join(summary_bits)

    item = {
        "source": "houston_permits:monthly",
        "source_url": HOUSTON_URL,
        "title": headline[:300],
        "summary": summary[:600],
        "raw_json": {
            "city_hint": "houston",
            "period": period.isoformat(),
            "prior_period": prior_period.isoformat(),
            "single_family": sf,
            "multi_family": mf,
            "residential": res,
            "single_family_prior": psf,
            "multi_family_prior": pmf,
            "residential_prior": pres,
            "mf_mom_pct": mf_pct,
            "res_mom_pct": res_pct,
            "data_age_days": age_days,
            "stale_data": stale,
            "published_date": period.isoformat(),
            "region": "usa",
            "country_code": "US",
            "msa_hint": "houston",
            "category_hint": "macro_indicator",
        },
        "dedup_key": _dedup_key(period.isoformat()),
    }
    return [item][:limit]
