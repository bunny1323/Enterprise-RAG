"""
Metadata Enrichment Service.
Loads industry-specific configuration and enriches chunks with domain metadata.
"""
import json
from pathlib import Path
from typing import Any

from app.config.logging import get_logger
from app.models.chunk import Chunk

logger = get_logger(__name__)

# Default metadata when no industry config is found
_DEFAULT_CONFIG: dict[str, Any] = {
    "domain": "generic",
    "entity_types": [],
    "relation_types": [],
    "security_classifications": ["PUBLIC", "INTERNAL", "RESTRICTED"],
    "chunking_rules": {
        "parent_token_range": [1024, 2048],
        "child_token_range": [128, 256],
    },
}


class MetadataService:
    """
    Stateless metadata enrichment service.

    Loads an industry JSON config and annotates chunks with domain-specific
    metadata: industry domain, access classification, and entity type hints.
    """

    def __init__(self, config_dir: str = "./config/industries") -> None:
        self._config_dir = Path(config_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def load_config(self, industry: str) -> dict[str, Any]:
        """
        Load industry configuration from JSON file.

        Args:
            industry: Industry name (matches filename stem, e.g. 'manufacturing').

        Returns:
            Configuration dict. Falls back to _DEFAULT_CONFIG if file not found.
        """
        if industry in self._cache:
            return self._cache[industry]

        config_path = self._config_dir / f"{industry}.json"
        if not config_path.exists():
            logger.warning(
                "metadata.config_not_found",
                industry=industry,
                searched=str(config_path),
            )
            self._cache[industry] = _DEFAULT_CONFIG.copy()
            return self._cache[industry]

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self._cache[industry] = config
            logger.info("metadata.config_loaded", industry=industry)
            return config
        except (json.JSONDecodeError, OSError) as err:
            logger.error("metadata.config_load_error", error=str(err), industry=industry)
            self._cache[industry] = _DEFAULT_CONFIG.copy()
            return self._cache[industry]

    def enrich_chunks(self, chunks: list[Chunk], industry: str) -> list[Chunk]:
        """
        Enrich chunks with industry-specific metadata.

        Enrichment adds:
        - industry_domain: from config
        - access_classification: default from config (INTERNAL)
        - entity_type_hints: list of potential entity types for this domain
        - chunking_rules: applied token range config

        Args:
            chunks: List of raw chunks from ChunkingService.
            industry: Industry domain identifier.

        Returns:
            Enriched chunk list (same objects, metadata mutated).
        """
        config = self.load_config(industry)
        default_classification = self._get_default_classification(config)
        entity_types = config.get("entity_types", [])
        relation_types = [r.get("name", "") for r in config.get("relation_types", [])]
        chunking_rules = config.get("chunking_rules", {})

        enriched: list[Chunk] = []
        for chunk in chunks:
            # Clone metadata dict to avoid mutation across chunks
            meta = dict(chunk.metadata)
            meta.update(
                {
                    "entity_type_hints": entity_types,
                    "relation_type_hints": relation_types,
                    "chunking_rules": chunking_rules,
                    "domain_config_version": config.get("version", "1.0"),
                }
            )
            enriched_chunk = chunk.model_copy(
                update={
                    "industry_domain": industry,
                    "access_classification": chunk.access_classification
                    if chunk.access_classification != "INTERNAL"
                    else default_classification,
                    "metadata": meta,
                }
            )
            enriched.append(enriched_chunk)

        logger.info(
            "metadata.enrichment_complete",
            industry=industry,
            chunks=len(enriched),
        )
        return enriched

    @staticmethod
    def _get_default_classification(config: dict[str, Any]) -> str:
        """Return the first classification from config, defaulting to INTERNAL."""
        classifications = config.get("security_classifications", [])
        # INTERNAL is always the safe default
        if "INTERNAL" in classifications:
            return "INTERNAL"
        return classifications[0] if classifications else "INTERNAL"
