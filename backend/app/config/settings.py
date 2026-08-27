"""
Application settings loaded from environment variables via Pydantic BaseSettings.
All configuration is centralized here — no hard-coded values elsewhere.
"""



from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="EnterpriseRAG")
    debug: bool = Field(default=False)
    port: int = Field(default=8000)

    # Observability
    otel_endpoint: str = Field(default="")  # e.g. http://localhost:4317
    otel_console: bool = Field(default=False)  # force console exporter in dev
    prometheus_port: int = Field(default=9090)

    # Voyage AI
    voyage_api_key: str = Field(...)
    voyage_model: str = Field(default="voyage-multimodal-3.5")
    voyage_text_model: str = Field(default="voyage-3.5")
    voyage_tpm_limit: int = Field(default=300_000)
    voyage_max_batch_size: int = Field(default=32)
    voyage_max_retries: int = Field(default=10)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    cache_ttl_seconds: int = Field(default=3600)

    # Ingestion Controls
    max_upload_size_mb: int = Field(default=100)
    ingestion_timeout_seconds: int = Field(default=1800)
    ingestion_timeout_validation: int = Field(default=30)
    ingestion_timeout_duplicate: int = Field(default=60)
    ingestion_timeout_parse: int = Field(default=900)
    ingestion_timeout_vision: int = Field(default=600)
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

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_vision_model: str = Field(default="llava:13b")

    # Local storage
    raw_storage_path: str = Field(default="./data/raw")
    processed_storage_path: str = Field(default="./data/processed")
    chunks_storage_path: str = Field(default="./data/chunks")

    @field_validator(
        "raw_storage_path",
        "processed_storage_path",
        "chunks_storage_path",
        mode="after",
    )
    @classmethod
    def ensure_paths_exist(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def database_url_async(self) -> str:
        return self.database_url


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings

    if _settings is None:
        _settings = Settings()

    return _settings