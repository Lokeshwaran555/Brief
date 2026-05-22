"""Generic Apify actor runner.

One wrapper used by every Apify-backed scout. Each source has its own
actor ID + input schema; callers build the input dict and we handle
auth, timeout, and error degradation consistently.

Silently returns [] when APIFY_TOKEN is not set.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from settings import settings

log = logging.getLogger(__name__)


def _sync_url(actor_id: str) -> str:
    # actor_id format: 'user~actor-name' (Apify API uses ~ not /)
    return f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"


async def run_actor(
    actor_id: str,
    run_input: dict[str, Any],
    *,
    timeout_seconds: int = 240,
) -> list[dict[str, Any]]:
    """Run an Apify actor synchronously and return the dataset items.

    Returns [] on any failure — never raises, so one flaky actor can't
    sink the ingest flow.
    """
    token = settings.apify_token
    if not token:
        log.info("apify: token not set — actor %s skipped", actor_id)
        return []

    try:
        async with httpx.AsyncClient(timeout=float(timeout_seconds + 10)) as client:
            resp = await client.post(
                _sync_url(actor_id),
                params={"token": token, "timeout": str(timeout_seconds)},
                json=run_input,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("apify actor %s failed: %s", actor_id, e)
        return []

    if not isinstance(data, list):
        log.warning("apify actor %s returned non-list: %s", actor_id, str(data)[:200])
        return []
    return data
