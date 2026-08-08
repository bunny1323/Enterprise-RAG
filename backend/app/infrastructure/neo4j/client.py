"""
Neo4j infrastructure client using the official async driver.
Creates document hierarchy graphs: Document → Section → Chunk.
"""
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config.logging import get_logger
from app.models.chunk import Chunk

logger = get_logger(__name__)


class Neo4jClient:
    """Async Neo4j client for document hierarchy graph operations."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the async driver (connection pool under the hood)."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            max_connection_lifetime=3600,
            max_connection_pool_size=10,
        )
        await self._driver.verify_connectivity()
        logger.info("neo4j.connected", uri=self._uri)

    async def close(self) -> None:
        """Close the driver and release all connections."""
        if self._driver:
            await self._driver.close()
            logger.info("neo4j.closed")

    # ── Schema ─────────────────────────────────────────────────────────────────

    async def init_schema(self) -> None:
        """Create uniqueness constraints for Document and Chunk nodes."""
        driver = self._require_driver()
        async with driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT unique_document_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )
        logger.info("neo4j.schema_initialized")

    # ── Graph operations ───────────────────────────────────────────────────────

    async def create_document_tree(
        self,
        document_id: str,
        chunks: list[Chunk],
    ) -> None:
        """
        Build the document hierarchy graph in Neo4j.

        Graph shape:
            (:Document {id})-[:HAS_SECTION]->(:Section {name})-[:CONTAINS_CHUNK]->(:Chunk {...})

        Parent chunks (parent_id is None) become Section nodes.
        Child chunks are linked under their parent Section.
        """
        driver = self._require_driver()

        async with driver.session() as session:
            # Upsert the root Document node
            await session.run(
                "MERGE (d:Document {id: $doc_id})",
                doc_id=document_id,
            )

            # Separate parents and children for two-phase insertion
            parent_chunks = [c for c in chunks if c.parent_id is None]
            child_chunks = [c for c in chunks if c.parent_id is not None]

            # Create Section nodes (parent chunks) linked to Document
            for parent in parent_chunks:
                props: dict[str, Any] = {
                    "doc_id": document_id,
                    "chunk_id": parent.chunk_id,
                    "page_number": parent.page_number,
                    "chunk_type": parent.chunk_type.value,
                    "hierarchy_path": parent.hierarchy_path,
                }
                await session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (s:Section {chunk_id: $chunk_id})
                    SET s.page_number = $page_number,
                        s.chunk_type  = $chunk_type,
                        s.hierarchy_path = $hierarchy_path
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    **props,
                )

            # Create Chunk nodes linked to their parent Section
            for child in child_chunks:
                child_props: dict[str, Any] = {
                    "doc_id": document_id,
                    "parent_id": child.parent_id,
                    "chunk_id": child.chunk_id,
                    "page_number": child.page_number,
                    "chunk_type": child.chunk_type.value,
                    "hierarchy_path": child.hierarchy_path,
                }
                await session.run(
                    """
                    MATCH (s:Section {chunk_id: $parent_id})
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.page_number    = $page_number,
                        c.chunk_type     = $chunk_type,
                        c.hierarchy_path = $hierarchy_path
                    MERGE (s)-[:CONTAINS_CHUNK]->(c)
                    """,
                    **child_props,
                )

        logger.info(
            "neo4j.document_tree_created",
            document_id=document_id,
            sections=len(parent_chunks),
            chunks=len(child_chunks),
        )

    async def verify_connectivity(self) -> bool:
        """Return True if the driver can reach the Neo4j server."""
        try:
            driver = self._require_driver()
            await driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ── Private helpers ────────────────────────────────────────────────────────

    def _require_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jClient.connect() must be called first")
        return self._driver
