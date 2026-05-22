"""Azure AD → Sobha MD Intelligence app client.

Service-principal (client credentials) flow via direct httpx calls —
no msal dep needed. Tokens cached in-memory with ~5 min safety buffer
before the stated expiry.

The integration gracefully no-ops when any of the required env vars
are unset, so other flows keep working while Azure wiring is in progress.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional
from urllib.parse import urljoin

import httpx

from settings import settings

log = logging.getLogger(__name__)


_TOKEN_CACHE: dict[str, Any] = {"access_token": None, "expires_at": 0}
# 2026-04-30 audit-round: serialize token refreshes so two concurrent
# callers don't both fetch a fresh token (cheap but wasteful + risks
# tenant rate-limit on the bootstrap call).
import asyncio as _aio
_TOKEN_LOCK = _aio.Lock()


def _enabled() -> bool:
    return bool(
        settings.azure_tenant_id
        and settings.azure_client_id
        and settings.azure_client_secret
        and settings.azure_api_base_url
    )


def _scope() -> str:
    if settings.azure_api_scope:
        return settings.azure_api_scope
    # App Service "Easy Auth" default: api://<app-client-id>/.default
    return f"api://{settings.azure_client_id}/.default"


def _base_url() -> str:
    raw = (settings.azure_api_base_url or "").rstrip("/")
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    return raw


async def _get_token(client: httpx.AsyncClient) -> str | None:
    """Client-credentials flow against the tenant's token endpoint.
    In-memory cache with 5-minute safety buffer; serialised under
    _TOKEN_LOCK so concurrent callers share one refresh.
    """
    if not _enabled():
        return None
    now = time.time()
    cached = _TOKEN_CACHE.get("access_token")
    if cached and _TOKEN_CACHE.get("expires_at", 0) - 300 > now:
        return cached
    async with _TOKEN_LOCK:
        # Re-check after acquiring lock — another coroutine may have
        # already refreshed while we waited.
        now = time.time()
        cached = _TOKEN_CACHE.get("access_token")
        if cached and _TOKEN_CACHE.get("expires_at", 0) - 300 > now:
            return cached
        url = f"https://login.microsoftonline.com/{settings.azure_tenant_id}/oauth2/v2.0/token"
        try:
            resp = await client.post(
                url,
                data={
                    "client_id": settings.azure_client_id,
                    "client_secret": settings.azure_client_secret,
                    "scope": _scope(),
                    "grant_type": "client_credentials",
                },
                timeout=15.0,
            )
            if resp.status_code != 200:
                log.warning("azure token failed: %s %s", resp.status_code, resp.text[:400])
                return None
            data = resp.json()
            tok = data.get("access_token")
            exp = now + float(data.get("expires_in", 3600))
            _TOKEN_CACHE["access_token"] = tok
            _TOKEN_CACHE["expires_at"] = exp
            return tok
        except Exception as e:
            log.warning("azure token exception: %s", e)
            return None


async def fetch(path: str, *, method: str = "GET", params: dict | None = None, json_body: dict | None = None) -> Optional[dict]:
    """Call an endpoint on the Sobha Intelligence app with bearer auth.

    Returns parsed JSON (dict or list), None on failure. Never raises
    so the investigation flow never blocks on MIS being down.
    """
    if not _enabled():
        return None
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        tok = await _get_token(client)
        if not tok:
            return None
        url = urljoin(_base_url() + "/", path.lstrip("/"))
        try:
            resp = await client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
            )
        except Exception as e:
            log.warning("azure fetch exception %s: %s", url, e)
            return None
        if resp.status_code >= 400:
            log.warning("azure fetch %s → %s %s", url, resp.status_code, resp.text[:300])
            return None
        ct = resp.headers.get("content-type", "")
        if "json" not in ct.lower():
            return {"_raw": resp.text[:2000], "_content_type": ct}
        try:
            return resp.json()
        except Exception:
            return None


async def _try_token(client: httpx.AsyncClient, scope: str) -> dict[str, Any]:
    """One attempt at client-credentials with an explicit scope.
    Returns {scope, status, error, error_description, access_token?}.
    """
    if not (settings.azure_tenant_id and settings.azure_client_id and settings.azure_client_secret):
        return {"scope": scope, "status": "skipped", "error": "missing creds"}
    url = f"https://login.microsoftonline.com/{settings.azure_tenant_id}/oauth2/v2.0/token"
    try:
        resp = await client.post(
            url,
            data={
                "client_id": settings.azure_client_id,
                "client_secret": settings.azure_client_secret,
                "scope": scope,
                "grant_type": "client_credentials",
            },
            timeout=15.0,
        )
    except Exception as e:
        return {"scope": scope, "status": "exception", "error": str(e)[:300]}
    out: dict[str, Any] = {"scope": scope, "status": resp.status_code}
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text[:400]}
    if resp.status_code == 200 and body.get("access_token"):
        out["access_token"] = body["access_token"]
        out["expires_in"] = body.get("expires_in")
    else:
        out["error"] = body.get("error")
        out["error_description"] = (body.get("error_description") or "")[:400]
        out["error_codes"] = body.get("error_codes")
    return out


async def diagnose() -> dict[str, Any]:
    """Report wiring status + try multiple scopes + probe endpoints.
    Surface via GET /debug/azure-intel.
    """
    out: dict[str, Any] = {
        "enabled": _enabled(),
        "tenant_id_set": bool(settings.azure_tenant_id),
        "client_id_set": bool(settings.azure_client_id),
        "client_secret_set": bool(settings.azure_client_secret),
        "base_url": _base_url() or None,
        "configured_scope": _scope() if settings.azure_client_id else None,
    }
    if not _enabled():
        out["message"] = "Missing env vars — set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_API_BASE_URL in Railway."
        return out

    # Try 5 different scopes so we can see which one this SP is actually
    # authorised for. DevOps-style SPs often only have ARM scope; Easy
    # Auth apps typically accept api://<app-id>/.default.
    cid = settings.azure_client_id or ""
    host = _base_url().replace("https://", "").replace("http://", "").rstrip("/")
    candidate_scopes = [
        _scope(),  # configured / default api://<cid>/.default
        f"{cid}/.default",  # modern format without api:// prefix
        "https://management.azure.com/.default",  # ARM (DevOps SPs)
        f"https://{host}/.default" if host else None,  # resource-as-scope
        "https://graph.microsoft.com/.default",  # Graph, sanity check
    ]
    candidate_scopes = [s for s in candidate_scopes if s]

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        token_attempts: list[dict[str, Any]] = []
        working_token: str | None = None
        working_scope: str | None = None
        for scope in candidate_scopes:
            res = await _try_token(client, scope)
            # Don't echo the full token in the diagnostic — last 6 chars only.
            if res.get("access_token"):
                tok = res.pop("access_token")
                res["token_tail"] = tok[-6:]
                if not working_token:
                    working_token = tok
                    working_scope = scope
            token_attempts.append(res)
        out["token_attempts"] = token_attempts
        out["auth"] = "ok" if working_token else "failed"
        if not working_token:
            out["message"] = (
                "All scope attempts failed. Likely causes: (a) secret wrong or expired, "
                "(b) this SP is not authorised as an API consumer for the Intelligence app "
                "(common for DevOps service connections — they have ARM scope only), "
                "(c) admin needs to add API permission / app role on the app registration "
                "that backs the MD Intelligence app. Check token_attempts for the exact error."
            )
            return out

        out["working_scope"] = working_scope

        # Probe common endpoints with the working token.
        probes = [
            "/",
            "/api",
            "/api/health",
            "/health",
            "/api/projects",
            "/api/projects/active",
            "/api/inventory",
            "/api/sales",
            "/api/properties",
            "/swagger/index.html",
            "/swagger",
            "/openapi.json",
            "/.well-known/openid-configuration",
        ]
        probe_results: list[dict[str, Any]] = []
        headers = {"Authorization": f"Bearer {working_token}", "Accept": "application/json"}
        for p in probes:
            url = urljoin(_base_url() + "/", p.lstrip("/"))
            try:
                r = await client.get(url, headers=headers)
                sample = ""
                ct = r.headers.get("content-type", "")
                if r.status_code < 400:
                    body = r.text[:400]
                    sample = body.replace("\n", " ")
                probe_results.append(
                    {
                        "path": p,
                        "status": r.status_code,
                        "content_type": ct,
                        "sample": sample if sample else None,
                    }
                )
            except Exception as e:
                probe_results.append({"path": p, "error": str(e)[:200]})
        out["probes"] = probe_results
    return out
