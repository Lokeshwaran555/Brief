from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_service_role_key: str

    # Optional — only used for embeddings (nv-embedqa-e5-v5) when set.
    # When unset, embed() returns None and pgvector dedup falls back to
    # title-hash. Chat goes through llm_* override (Groq) regardless.
    nvidia_api_key: str | None = None
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    nvidia_embed_dims: int = 1024
    signal_dedup_similarity: float = 0.86
    signal_dedup_window_days: int = 30

    # Optional chat-LLM override. When set, all chat_json + CrewAI calls
    # route through this provider instead of NIM. Embeddings keep using
    # NIM (Groq doesn't offer embeddings; nvidia_embed_model stays
    # 1024-dim). Recommended values for Groq:
    #   llm_base_url = "https://api.groq.com/openai/v1"
    #   llm_api_key  = <Groq key>
    #   llm_model    = "llama-3.3-70b-versatile"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    # Optional cheaper model for the classifier. Classifier fires
    # 10-30× per ingest run (one call per raw signal); on Groq's free
    # tier this dominates the daily token budget. Routing it through
    # llama-3.1-8b-instant (5× higher TPD on free tier) preserves the
    # daily budget for the heavier brief + CEO synth tasks.
    # Recommended:
    #   llm_classifier_model = "llama-3.1-8b-instant"
    # Falls back to llm_model when unset.
    llm_classifier_model: str | None = None

    # Optional CrewAI-specific model. CrewAI orchestration uses
    # litellm under the hood and breaks against reasoning-style models
    # (gpt-oss-120b's reasoning_content channel returns 'None or empty
    # response' to the litellm content extractor). Direct flows can use
    # gpt-oss-120b safely; CrewAI flows need a non-reasoning model.
    # Recommended for Groq:
    #   llm_crew_model = "llama-3.3-70b-versatile"
    # Falls back to llm_model when unset.
    llm_crew_model: str | None = None

    ingest_interval_minutes: int = 15
    intel_query_limit: int = 40
    intel_concurrency: int = 4

    port: int = 8080
    log_level: str = "INFO"

    # Langfuse Cloud (observability). Leave keys unset to disable tracing.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Apify (Instagram + Bayut + Dubizzle + on-demand YT). Leave unset to disable.
    apify_token: str | None = None

    # Tavily (web search for the investigation layer). Leave unset to disable
    # deep research; other flows keep working.
    tavily_api_key: str | None = None

    # Azure AD service principal → Sobha MD Intelligence app API.
    # Used by the investigation crew's sobha_context step to compare
    # competitor signals against Sobha's own active projects. Leave any
    # field unset to disable the integration.
    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_client_secret: str | None = None
    azure_api_base_url: str | None = None       # full origin, no trailing slash
    azure_api_scope: str | None = None          # default: api://<client_id>/.default


settings = Settings()
