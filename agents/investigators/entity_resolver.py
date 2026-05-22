"""Pin down the entities behind a seed signal.

Single-shot NVIDIA NIM call (not a Crew — this is a cheap classification
step, no reasoning chain needed). Output feeds the spider's query and
the investigation tables.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tools.nvidia_llm import chat_json

log = logging.getLogger(__name__)


DEV_SLUGS = [
    "emaar", "damac", "aldar", "nakheel", "modon", "dubaih",
    "binghatti", "azizi", "ellington", "omniyat", "deyaar",
    "meraas", "sobha",
]


SYSTEM = """You resolve the entities behind a Dubai real-estate market signal so an
investigation agent can search for more information about it.

Given a seed signal (headline + dek + raw source), return the key
entities as JSON. Leave fields null if genuinely unknown — DO NOT guess.

Event types are dynamic — pick the best label for what happened. Common
types: launch, phase-release, price-change, payment-plan-change, partnership,
joint-venture, land-acquisition, financing, sukuk, commission-move,
handover-delay, regulatory, hiring, leadership-move, expansion,
award, rumor, other. You may invent a new type if none fit.

Respond with JSON ONLY:
{
  "event_type": "...",
  "developer": "Emaar Properties" | null,
  "developer_slug": "emaar" | null,  // use one of: emaar, damac, aldar, nakheel, modon, dubaih, binghatti, azizi, ellington, omniyat, deyaar, meraas, sobha, or null
  "project": "Creek Harbour — Phase 5" | null,
  "location": "Dubai Creek Harbour" | null,
  "event_date": "2026-04-24" | null,
  "partners": ["Mubadala"] | [],
  "confidence": 0.0
}"""


async def resolve(signal: dict[str, Any]) -> dict[str, Any]:
    """Takes a signal row (from the `signals` table). Returns entity dict.

    Two-attempt retry pattern — NVIDIA NIM occasionally rate-limits
    under concurrent load (auto-investigation fans out multiple jobs)
    or drops JSON mode and returns prose. The retry with an explicit
    'start with {' nudge catches both cases.
    """
    payload = {
        "headline": signal.get("headline"),
        "dek": signal.get("dek"),
        "dev_slug_hint": signal.get("dev_slug"),
        "entities_hint": signal.get("entities") or [],
        "category": signal.get("category"),
        "decision_tag": signal.get("decision_tag"),
        "urls": (signal.get("urls") or [])[:3],
    }
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Seed signal:\n{json.dumps(payload, indent=2)}"},
    ]
    parsed: dict[str, Any] = {}
    try:
        parsed = await chat_json(messages, max_tokens=500)
    except Exception as e1:
        log.warning("entity_resolver attempt 1 failed: %s — retrying", e1)
        retry_messages = messages + [
            {
                "role": "user",
                "content": "Return ONLY the JSON object. No prose. Start your response with '{'.",
            }
        ]
        try:
            parsed = await chat_json(retry_messages, max_tokens=500)
        except Exception as e2:
            log.warning("entity_resolver attempt 2 failed: %s", e2)
            parsed = {}

    # Seed headline + dev_slug hint as fallbacks so downstream spider
    # has SOMETHING to work with even when the LLM fails completely.
    if not parsed.get("headline"):
        parsed["headline"] = signal.get("headline")
    if not parsed.get("developer_slug") and signal.get("dev_slug"):
        parsed["developer_slug"] = signal.get("dev_slug")
    if not parsed.get("developer") and signal.get("dev_slug"):
        parsed["developer"] = signal.get("dev_slug")

    # Normalize slug to known list (defense in depth).
    slug = (parsed.get("developer_slug") or "").lower().strip() or None
    if slug and slug not in DEV_SLUGS:
        slug = None
    parsed["developer_slug"] = slug
    return parsed
