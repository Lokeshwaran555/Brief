"""Intel scorer — wraps nvidia_llm.classify_intel with bounded concurrency.

Takes raw intel_raw rows (already in Supabase), calls NVIDIA NIM for each,
returns list of intel_scored payloads ready for insert.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from settings import settings
from tools.nvidia_llm import classify_intel

log = logging.getLogger(__name__)


def _tidy_score(raw: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    """Normalise LLM output. Region/country are kept on the dict but
    NOT persisted to intel_scored (that table doesn't have those
    columns and isn't worth a schema change — the values get used
    in-memory by ingest_flow._promote_to_signals before being dropped
    on insert)."""
    return {
        "raw_id": raw["id"],
        "keep": bool(score.get("keep", False)),
        "category": (score.get("category") or "")[:40] or None,
        # 2026-04-30 round 7: LLM-driven subcategory routing. See
        # tools/nvidia_llm.py:INTEL_SYSTEM for the taxonomy. Themed-page
        # endpoints filter by (category, subcategory) instead of keyword
        # post-filters.
        "subcategory": (score.get("subcategory") or "")[:40] or None,
        "decision_tag": (score.get("decisionTag") or "")[:40] or None,
        "priority": (score.get("priority") or "")[:10] or None,
        "confidence": float(score.get("confidence") or 0.0),
        "headline": (score.get("headline") or raw.get("title") or "")[:200],
        "dek": (score.get("dek") or "")[:400],
        "entities": score.get("entities") or [],
        "reason": (score.get("reason") or "")[:200],
        "model": settings.nvidia_model,
        # Carried through to _promote_to_signals; stripped before insert.
        "region": (score.get("region") or "").strip().lower() or None,
        "country_code": (score.get("country_code") or "").strip().upper()[:2] or None,
    }


async def _score_one(
    raw: dict[str, Any], sem: asyncio.Semaphore
) -> dict[str, Any] | None:
    signal = {
        "source": raw.get("source"),
        "title": raw.get("title"),
        "summary": raw.get("summary"),
        "url": raw.get("source_url"),
        "date": (raw.get("raw_json") or {}).get("date"),
    }
    async with sem:
        try:
            score = await classify_intel(signal)
        except Exception as e:
            log.warning("classify failed for raw_id=%s: %s", raw.get("id"), e)
            return {
                "raw_id": raw["id"],
                "keep": False,
                "reason": f"llm-error: {str(e)[:120]}",
                "model": settings.nvidia_model,
            }
    return _tidy_score(raw, score)


async def score_batch(
    raws: list[dict[str, Any]],
    *,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    if not raws:
        return []
    sem = asyncio.Semaphore(concurrency or settings.intel_concurrency)
    results = await asyncio.gather(*[_score_one(r, sem) for r in raws])
    return [r for r in results if r]
