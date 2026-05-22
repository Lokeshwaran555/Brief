"""Source-publish-date parser + age-based priority decay.

Scouts capture publish dates under different keys depending on the
source:
  - gnews/forum/reddit/youtube RSS: raw_json.date (RFC-822 or ISO)
  - reddit .json:                    raw_json.created_utc (unix float)
  - Apify Instagram:                 raw_json.timestamp (ISO)
  - Tavily:                          raw_json.published_date
  - Apify YouTube search:            raw_json.date (ISO)

This module normalises all of those to a single aware UTC datetime and
decays the classifier's priority based on how old the content actually
is. Fixes the "47-week-old IG post showing JUST NOW · priority=high"
problem we hit on day 1.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


_DATE_KEYS = (
    "published_at",
    "published_date",
    "date",
    "timestamp",
    "created_utc",
    "source_published_at",
)


def parse_published_at(raw: dict[str, Any] | None) -> datetime | None:
    """Return the source publish date as aware UTC, or None.

    Accepts the raw `intel_raw` row OR just its `raw_json` sub-dict.
    Tries the full set of date field names used across scouts.
    """
    if not raw:
        return None
    rj = raw.get("raw_json") if "raw_json" in raw else raw
    if not isinstance(rj, dict):
        return None

    for key in _DATE_KEYS:
        val = rj.get(key)
        if val is None or val == "":
            continue
        dt = _coerce_datetime(val)
        if dt:
            return dt
    return None


def _coerce_datetime(val: Any) -> datetime | None:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        # Unix timestamp. Reject obviously-bogus values.
        if val < 946684800:  # before 2000-01-01
            return None
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        # Try common formats first — cheaper than dateutil when possible.
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(val.replace("Z", "+0000"), fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # Fall back to dateutil (handles RFC-822, etc.)
        try:
            from dateutil import parser
            dt = parser.parse(val)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


_PRI_DOWNGRADE = {"high": "medium", "medium": "low", "low": "low"}


def age_days(published_at: datetime | None) -> int | None:
    if not published_at:
        return None
    return max(0, (datetime.now(timezone.utc) - published_at).days)


def decay_priority(
    priority: str | None,
    published_at: datetime | None,
    *,
    fresh_days: int = 7,
    stale_days: int = 45,
) -> str | None:
    """Knock priority down as content ages, drop if truly stale.

    Returns the decayed priority, or None to signal 'reject this signal'.
    Unknown publish dates are treated as fresh (can't penalise what we
    can't measure).

    Tightened from the original (14d / 180d) — a year-old LinkedIn post
    re-surfacing as 'medium priority intel' is the failure mode this
    fixes. After 7d a story is no longer fresh; after 45d it is no
    longer market-relevant for a daily MD brief.
    """
    if priority is None:
        return None
    days = age_days(published_at)
    if days is None:
        return priority
    if days <= fresh_days:
        return priority
    if days > stale_days:
        log.info("freshness: rejecting signal (age=%dd > %dd): %s", days, stale_days, priority)
        return None  # reject
    # In between → downgrade one tier.
    decayed = _PRI_DOWNGRADE.get((priority or "").lower(), priority)
    if decayed != priority:
        log.info("freshness: decaying %s → %s (age=%dd)", priority, decayed, days)
    return decayed
