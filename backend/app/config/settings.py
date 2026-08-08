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

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = Field(default="EnterpriseRAG", description="Application display name")
    debug: bool = Field(default=False, description="Enable debug mode")
    port: int = Field(default=8000, description="Server port")

    # ── Voyage AI ──────────────────────────────────────────────────────────────
    voyage_api_key: str = Field(..., description="Voyage AI API key for embeddings")
    voyage_model: str = Field(
        default="voyage-multimodal-3.5", description="Voyage multimodal embedding model name"
    )
    voyage_text_model: str = Field(
        default="voyage-3.5", description="Voyage text embedding model name"
    )

    # ── Weaviate Cloud ─────────────────────────────────────────────────────────
    weaviate_url: str = Field(..., description="Weaviate Cloud cluster URL")
    weaviate_api_key: str = Field(..., description="Weaviate Cloud API key")

    # ── Supabase PostgreSQL ────────────────────────────────────────────────────
    database_url: str = Field(..., description="PostgreSQL connection URL (asyncpg format)")

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    # ── Ollama (local vision) ──────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    ollama_vision_model: str = Field(
        default="llava:13b", description="Ollama vision model name"
    )

    # ── Local Storage ──────────────────────────────────────────────────────────
    raw_storage_path: str = Field(default="./data/raw", description="Raw upload storage path")
    processed_storage_path: str = Field(
        default="./data/processed", description="Processed document storage path"
    )
    chunks_storage_path: str = Field(
        default="./data/chunks", description="Chunk text storage path"
    )

    @field_validator("raw_storage_path", "processed_storage_path", "chunks_storage_path", mode="after")
    @classmethod
    def ensure_paths_exist(cls, v: str) -> str:
        """Auto-create storage directories on startup."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def database_url_async(self) -> str:
        """Return asyncpg-compatible URL (replace postgresql:// with postgres://)."""
        return self.database_url.replace("postgresql://", "postgresql://", 1)


# Module-level singleton — import `get_settings()` from dependencies for testability
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached Settings instance (loaded once at startup)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
