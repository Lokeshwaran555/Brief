"""CrewAI LLM adapter for NVIDIA NIM.

CrewAI uses litellm under the hood. NVIDIA's NIM is OpenAI-compatible,
so we register it as an OpenAI-style endpoint with a custom base_url.

Import `nvidia_llm` anywhere and pass to Agent(llm=nvidia_llm).
"""
from __future__ import annotations

import os

from crewai import LLM

from settings import settings


# litellm's native Groq adapter reads `GROQ_API_KEY` from the env in
# addition to the explicit `api_key` arg. Mirror our llm_api_key into
# that env var when Groq is the active provider so every internal
# litellm path (CrewAI agents, tool callbacks, retry layers) finds it.
if (settings.llm_base_url and "groq.com" in settings.llm_base_url.lower()
        and settings.llm_api_key and not os.environ.get("GROQ_API_KEY")):
    os.environ["GROQ_API_KEY"] = settings.llm_api_key


# Use the llm_* override (e.g. Groq) when set, else fall back to NIM.
# CrewAI talks via litellm. For Groq we use litellm's native `groq/`
# provider — `openai/` prefix with a Groq base_url 404s because
# litellm rewrites the URL path internally and hits an endpoint that
# doesn't exist on Groq. Other OpenAI-compatible providers stay on
# `openai/` + custom base_url, which still works.
_use_override = bool(settings.llm_base_url and settings.llm_api_key and settings.llm_model)


# CrewAI-specific model override. Reasoning models (gpt-oss-120b et al.)
# produce empty content via litellm's CrewAI integration because their
# output lives in `reasoning_content`. The crew model must be a plain
# completion model (llama-3.3-70b-versatile is the proven default).
# Direct flows (classifier, CEO scan, pattern prose) keep using
# settings.llm_model — they don't go through litellm.
_crew_model = settings.llm_crew_model or settings.llm_model


def _build_llm() -> LLM:
    if _use_override:
        is_groq = "groq.com" in (settings.llm_base_url or "").lower()
        if is_groq:
            return LLM(
                model=f"groq/{_crew_model}",
                api_key=settings.llm_api_key,
                temperature=0.3,
                max_tokens=1500,
            )
        return LLM(
            model=f"openai/{_crew_model}",
            base_url=settings.llm_base_url.rstrip("/"),
            api_key=settings.llm_api_key,
            temperature=0.3,
            max_tokens=1500,
        )
    if settings.nvidia_api_key:
        return LLM(
            model=f"openai/{settings.nvidia_model}",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=settings.nvidia_api_key,
            temperature=0.3,
            max_tokens=1500,
        )
    raise RuntimeError(
        "No chat LLM configured. Set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL "
        "(recommended: Groq), or set NVIDIA_API_KEY."
    )


nvidia_llm = _build_llm()
