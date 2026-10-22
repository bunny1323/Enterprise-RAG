"""
conftest.py for Enterprise-RAG backend unit tests.

Provides environment variable stubs so that `app.config.settings.Settings`
can be instantiated in the CI environment without real credentials.
All external service calls (PostgreSQL, Weaviate, Neo4j, Redis) are mocked
at the individual test level using unittest.mock.
"""
import os

# ── Minimal stubs so pydantic-settings can instantiate Settings ────────────────
# These are NOT real credentials. Integration tests that actually connect to
# external services are marked @pytest.mark.integration and skipped in CI.
_CI_ENV = {
    "APP_ENV": "test",
    "API_KEY_AUTH_ENABLED": "false",
    "DATABASE_URL": "postgresql://ci_user:ci_pass@localhost:5432/ci_db",
    "WEAVIATE_URL": "https://ci-stub.weaviate.network",
    "WEAVIATE_API_KEY": "ci-weaviate-key",
    "NEO4J_PASSWORD": "ci-neo4j-password",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "GROQ_API_KEY": "",
    "REDIS_URL": "",
    "LLM_PROVIDER": "groq",
    "LLM_MODEL": "llama-3.3-70b-versatile",
    "EMBEDDING_PROVIDER": "local",
    "VISION_PROVIDER": "disabled",
}

for key, value in _CI_ENV.items():
    os.environ.setdefault(key, value)
