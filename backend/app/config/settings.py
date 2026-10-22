"""
Application settings loaded from environment variables via Pydantic BaseSettings.
All configuration is centralized here — no hard-coded values elsewhere.
"""

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="EnterpriseRAG")
    app_env: str = Field(
        default="development",
        validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT", "ENV"),
    )
    debug: bool = Field(default=False)
    port: int = Field(default=8000)

    # Observability
    otel_endpoint: str = Field(default="")  # e.g. http://localhost:4317
    otel_console: bool = Field(default=False)  # force console exporter in dev
    prometheus_port: int = Field(default=9090)

    embedding_provider: str = Field(default="local")

    # Local embedding (sentence-transformers, CPU, no API key required)
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    local_embedding_device: str = Field(default="cpu")
    local_embedding_batch_size: int = Field(default=8)

    # Redis (Render Key Value in prod; localhost fallback only in local dev)
    redis_url: str = Field(default="")
    cache_ttl_seconds: int = Field(default=3600)

    # Security & Access Controls
    api_key_auth_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("API_KEY_AUTH_ENABLED", "AUTH_ENABLED"),
    )
    api_auth_token: str = Field(
        default="",
        validation_alias=AliasChoices("API_AUTH_TOKEN", "API_KEY"),
    )
    admin_api_token: str = Field(
        default="",
        validation_alias=AliasChoices("ADMIN_API_TOKEN", "ADMIN_KEY"),
    )
    api_tokens_json: str = Field(
        default="",
        validation_alias=AliasChoices("API_TOKENS_MAPPING", "API_TOKENS_JSON"),
    )
    default_tenant_id: str = Field(
        default="default",
        validation_alias=AliasChoices("DEFAULT_TENANT_ID"),
    )
    trusted_proxies: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "::1"],
        validation_alias=AliasChoices("TRUSTED_PROXIES"),
    )
    pii_detection_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("PII_DETECTION_ENABLED"),
    )

    # Ingestion Controls
    max_upload_size_mb: int = Field(default=100)
    ingestion_timeout_seconds: int = Field(default=1800)
    ingestion_timeout_validation: int = Field(default=30)
    ingestion_timeout_duplicate: int = Field(default=60)
    ingestion_timeout_parse: int = Field(default=900)
    ingestion_timeout_vision: int = Field(default=7200)
    ingestion_timeout_chunk: int = Field(default=300)
    ingestion_timeout_incremental: int = Field(default=300)
    ingestion_timeout_metadata: int = Field(default=120)
    ingestion_timeout_embedding: int = Field(default=600)
    ingestion_timeout_indexing: int = Field(default=600)
    parse_profile: str = Field(default="BALANCED")  # FAST | BALANCED | HIGH_ACCURACY

    # Weaviate
    weaviate_url: str = Field(...)
    weaviate_api_key: str = Field(...)

    # PostgreSQL / Supabase
    database_url: str = Field(...)

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(...)

    # Vision Provider ("disabled" for cloud/Render; "gemini" for cloud ingestion)
    vision_provider: str = Field(
        default="disabled",
        validation_alias=AliasChoices("VISION_PROVIDER", "VISION_SERVICE_PROVIDER"),
    )
    vision_max_concurrency: int = Field(
        default=1,
        gt=0,
        validation_alias=AliasChoices("VISION_MAX_CONCURRENCY"),
    )
    vision_request_timeout: float = Field(
        default=600.0,
        gt=0,
        validation_alias=AliasChoices("VISION_REQUEST_TIMEOUT"),
    )
    vision_max_retries: int = Field(
        default=3,
        ge=0,
        validation_alias=AliasChoices("VISION_MAX_RETRIES"),
    )
    vision_min_request_interval: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("VISION_MIN_REQUEST_INTERVAL"),
    )
    vision_max_image_dimension: int = Field(
        default=1024,
        gt=0,
        validation_alias=AliasChoices("VISION_MAX_IMAGE_DIMENSION"),
    )
    vision_cache_dir: str = Field(
        default="./data/processed/vision_cache",
        validation_alias=AliasChoices("VISION_CACHE_DIR"),
    )
    ingestion_timeout_vision_override: int = Field(
        default=7200,
        gt=0,
        validation_alias=AliasChoices("INGESTION_TIMEOUT_VISION"),
    )

    # Gemini Cloud Vision (primary cloud provider, VISION_PROVIDER=gemini)
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY"),
    )
    gemini_vision_model: str = Field(
        default="gemini-3.8-flash",
        validation_alias=AliasChoices("GEMINI_VISION_MODEL"),
    )

    # Ollama (optional local-dev fallback for LLM)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")
    ollama_timeout: float = Field(default=120.0, gt=0, validation_alias=AliasChoices("OLLAMA_TIMEOUT", "LLM_FALLBACK_TIMEOUT"))

    # Generation LLM. Groq is Primary, Ollama is Fallback.
    llm_provider: str = Field(default="groq")
    llm_model: str = Field(default="llama-3.3-70b-versatile", validation_alias=AliasChoices("LLM_MODEL", "GROQ_MODEL"))
    llm_base_url: str = Field(default="", validation_alias=AliasChoices("LLM_BASE_URL", "GROQ_BASE_URL"))
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "LLM_API_KEY"),
    )
    llm_timeout: float = Field(default=30.0, gt=0, validation_alias=AliasChoices("LLM_TIMEOUT", "LLM_TIMEOUT_SECONDS"))
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=768, gt=0)

    # Local storage
    raw_storage_path: str = Field(default="./data/raw")
    processed_storage_path: str = Field(default="./data/processed")
    chunks_storage_path: str = Field(default="./data/chunks")

    @field_validator(
        "raw_storage_path",
        "processed_storage_path",
        "chunks_storage_path",
        "vision_cache_dir",
        mode="after",
    )
    @classmethod
    def ensure_paths_exist(cls, v: str) -> str:
        resolved = str(Path(v).resolve())
        Path(resolved).mkdir(parents=True, exist_ok=True)
        return resolved

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_mode(cls, value: object) -> object:
        """Accept common deployment-mode values while retaining a boolean setting."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value

    @field_validator(
        "neo4j_uri",
        "neo4j_user",
        "neo4j_password",
        "redis_url",
        "llm_model",
        "llm_base_url",
        "weaviate_url",
        mode="before",
    )
    @classmethod
    def strip_config_strings(cls, value: object) -> object:
        """Strip whitespace and accidental enclosing quotes from string configs."""
        if isinstance(value, str):
            return value.strip().strip("'\"")
        return value

    @model_validator(mode="after")
    def resolve_redis_fallback(self) -> "Settings":
        """Fall back to localhost Redis ONLY during local development (debug=True)."""
        if not self.redis_url and self.debug:
            self.redis_url = "redis://localhost:6379"
        return self

    @property
    def is_production(self) -> bool:
        """Return True if running in production mode."""
        env = (self.app_env or "").strip().lower()
        if env in {"production", "prod", "release", "staging"}:
            return True
        if not self.debug and env not in {"development", "dev", "test", "testing", "local"}:
            return True
        return False

    @model_validator(mode="after")
    def validate_startup_config(self) -> "Settings":
        """
        Startup configuration validation.

        Detects placeholder values, known-bad model IDs, and missing required
        credentials. Raises ValueError with an actionable message so the process
        fails fast instead of silently degrading at runtime.

        Critical checks raise ValueError (process will not start).
        Warning checks log instead of raising.
        """
        import logging
        _log = logging.getLogger(__name__)

        # ── 0. Production Security Enforcement ────────────────────────────────
        _PLACEHOLDER_TOKENS = (
            "<placeholder>", "<your-", "<password>", "<host>", "<port>",
            "<redis_password>", "<redis_host>", "<redis_port>",
            "your-key-here", "changeme",
        )

        def _has_placeholder(value: str) -> bool:
            v = value.lower()
            return any(tok in v for tok in _PLACEHOLDER_TOKENS)

        if self.is_production:
            if not self.api_key_auth_enabled:
                raise ValueError(
                    f"Production configuration error: API_KEY_AUTH_ENABLED must be set to 'true' "
                    f"in production mode (APP_ENV={self.app_env!r})."
                )
            if not self.api_auth_token and not self.admin_api_token and not self.api_tokens_json:
                raise ValueError(
                    "Production configuration error: API_AUTH_TOKEN, ADMIN_API_TOKEN, or API_TOKENS_MAPPING "
                    "must be configured when authentication is enabled in production mode."
                )
            if self.api_auth_token and _has_placeholder(self.api_auth_token):
                raise ValueError("Production configuration error: API_AUTH_TOKEN contains placeholder text.")
            if self.admin_api_token and _has_placeholder(self.admin_api_token):
                raise ValueError("Production configuration error: ADMIN_API_TOKEN contains placeholder text.")

        # ── 1. Placeholder detection ──────────────────────────────────────────

        if self.redis_url and _has_placeholder(self.redis_url):
            _log.warning(
                "startup.config_warning: REDIS_URL contains placeholder text — "
                "semantic cache disabled until a real URL is provided."
            )
            self.redis_url = ""  # treat as unconfigured

        # ── 2. Known-bad Gemini vision model names ────────────────────────────
        _KNOWN_BAD_VISION_MODELS = {
            "gemini-3-flash",
            "gemini-3.0-flash",
        }
        if (
            getattr(self, "vision_provider", "disabled") == "gemini"
            and self.gemini_vision_model in _KNOWN_BAD_VISION_MODELS
        ):
            raise ValueError(
                f"GEMINI_VISION_MODEL={self.gemini_vision_model!r} is not a valid "
                "Google AI model ID.  Supported models include: "
                "gemini-2.5-flash, gemini-1.5-flash, gemini-1.5-pro.  "
                "Update GEMINI_VISION_MODEL in your .env file."
            )

        # ── 3. Known-bad LLM model names ──────────────────────────────────────
        _KNOWN_BAD_LLM_MODELS = {
            "qwen/qwen3.8-27b",   # invalid Groq model ID
            "qwen3.8-27b",
            "qwen/qwen3.8",
        }
        if self.llm_provider == "groq" and self.llm_model in _KNOWN_BAD_LLM_MODELS:
            raise ValueError(
                f"LLM_MODEL={self.llm_model!r} is not a valid Groq model ID.  "
                "Valid Groq models include: llama-3.3-70b-versatile, "
                "llama-3.1-8b-instant, gemma2-9b-it.  "
                "Update LLM_MODEL in your .env file.  "
                "List all available models: curl -H 'Authorization: Bearer $GROQ_API_KEY' "
                "https://api.groq.com/openai/v1/models"
            )

        # ── 4. Missing Groq API key when provider=groq ────────────────────────
        if self.llm_provider == "groq" and not self.llm_api_key:
            _log.warning(
                "startup.config_warning: GROQ_API_KEY is empty — "
                "Groq generation will fail; Ollama fallback will be attempted."
            )

        # ── 5. Missing Gemini API key when vision=gemini ──────────────────────
        if getattr(self, "vision_provider", "disabled") == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when VISION_PROVIDER=gemini. "
                "Add GEMINI_API_KEY to your .env file."
            )

        return self

    @property
    def database_url_async(self) -> str:
        return self.database_url


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _settings

    if _settings is None or reload:
        _settings = Settings()

    return _settings
