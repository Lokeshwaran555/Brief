"""Langfuse v3 tracing wiring.

Langfuse Cloud projects on the new ingestion path require SDK v3+
("Faster Langfuse experience"). v2 traces appear to silently drop.

v3 uses get_client() (env-driven) and an @observe decorator that supports
both sync and async functions.

Idempotent. Silently no-ops when keys are not set.
"""
from __future__ import annotations

import logging
import os

from settings import settings

log = logging.getLogger(__name__)

_initialized = False
_enabled = False


def init() -> bool:
    """Configure Langfuse v3 client + verify auth. Returns True if traces will flow."""
    global _initialized, _enabled
    if _initialized:
        return _enabled

    pub = settings.langfuse_public_key
    sec = settings.langfuse_secret_key
    if not pub or not sec:
        log.info("langfuse: keys not set — tracing disabled")
        _initialized = True
        return False

    # v3 SDK reads creds purely from env. Make sure they're there even if
    # settings was loaded from a non-env source (e.g. .env file).
    os.environ["LANGFUSE_PUBLIC_KEY"] = pub
    os.environ["LANGFUSE_SECRET_KEY"] = sec
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host

    try:
        from langfuse import get_client
    except ImportError as e:
        log.warning("langfuse v3 not installed (%s) — tracing disabled", e)
        _initialized = True
        return False

    try:
        client = get_client()
        if not client.auth_check():
            log.warning("langfuse: auth_check failed — tracing disabled")
            _initialized = True
            return False
        log.info("langfuse v3: tracing enabled → %s", settings.langfuse_host)
        _initialized = True
        _enabled = True
        return True
    except Exception as e:
        log.warning("langfuse init failed: %s — tracing disabled", e)
        _initialized = True
        return False


def observe(name: str):
    """Decorator that records one Langfuse trace per call.

    v3's @observe lives at the package root and natively supports async.
    No-op fallback when langfuse isn't installed or keys are missing.
    """
    try:
        from langfuse import observe as _observe
        return _observe(name=name)
    except Exception:
        def _noop(fn):
            return fn
        return _noop


def flush() -> None:
    """Force-flush buffered spans. Call after long-running async work
    where the process might idle before the auto-flush timer fires."""
    if not _enabled:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        log.debug("langfuse flush failed: %s", e)
