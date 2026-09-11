"""
Neo4j infrastructure client using the official async driver.
Creates document hierarchy graphs with tenant isolation and stable graph schemas:
Document -> Section -> Chunk
Document -> SUPERSEDES -> Document
Chunk -> MENTIONS -> Entity
"""
import asyncio
import time
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
        self._schema_initialized: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self, retries: int = 3, initial_delay: float = 1.0) -> None:
        """Open the async driver and verify connectivity with bounded retry."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_lifetime=600,
                max_connection_pool_size=10,
                liveness_check_timeout=60,
                max_transaction_retry_time=30,
                keep_alive=True,
                connection_timeout=15.0,
            )

        last_err: Exception | None = None
        delay = initial_delay
        for attempt in range(1, retries + 1):
            try:
                await self._driver.verify_connectivity()
                logger.info("neo4j.connected", uri=self._uri, attempt=attempt)
                return
            except Exception as err:
                last_err = err
                logger.warning(
                    "neo4j.connect_retry",
                    attempt=attempt,
                    max_retries=retries,
                    error=str(err),
                )
                if attempt < retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 4.0)

        if last_err:
            raise last_err

    async def close(self) -> None:
        """Close the driver and release all connections."""
        if self._driver:
            try:
                res = self._driver.close()
                if asyncio.iscoroutine(res):
                    await res
            finally:
                self._driver = None
                self._schema_initialized = False
            logger.info("neo4j.closed")

    # ── Schema ─────────────────────────────────────────────────────────────────

    async def init_schema(self) -> None:
        """Create uniqueness constraints and indexes."""
        if self._schema_initialized:
            return

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
            await session.run(
                "CREATE INDEX idx_chunk_tenant IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.tenant_id, c.knowledge_base_id)"
            )
        self._schema_initialized = True
        logger.info("neo4j.schema_initialized")

    # ── Graph operations ───────────────────────────────────────────────────────

    async def create_document_tree(
        self,
        document_id: str,
        chunks: list[Chunk],
    ) -> None:
        """
        Build the document hierarchy graph in Neo4j with tenant isolation.

        Graph shape:
            (:Document)-[:HAS_SECTION]->(:Section)
            (:Section)-[:CONTAINS_CHUNK]->(:Chunk)
        """
        driver = self._require_driver()

        if not chunks:
            return

        tenant_id = chunks[0].tenant_id
        kb_id = chunks[0].knowledge_base_id

        parent_chunks = [c for c in chunks if c.parent_id is None]
        child_chunks = [c for c in chunks if c.parent_id is not None]

        sections = [
            {
                "chunk_id": c.chunk_id,
                "tenant_id": c.tenant_id,
                "kb_id": c.knowledge_base_id,
                "page_number": c.page_number,
                "chunk_type": c.chunk_type.value,
                "hierarchy_path": c.hierarchy_path,
            }
            for c in parent_chunks
        ]

        children = [
            {
                "parent_id": c.parent_id,
                "chunk_id": c.chunk_id,
                "tenant_id": c.tenant_id,
                "kb_id": c.knowledge_base_id,
                "page_number": c.page_number,
                "chunk_type": c.chunk_type.value,
                "hierarchy_path": c.hierarchy_path,
            }
            for c in child_chunks
        ]

        async def _write_tree(tx):
            await tx.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.tenant_id = $tenant_id,
                    d.knowledge_base_id = $kb_id
                """,
                doc_id=document_id,
                tenant_id=tenant_id,
                kb_id=kb_id,
            )

            if sections:
                await tx.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    UNWIND $sections AS section

                    MERGE (s:Section {chunk_id: section.chunk_id})
                    SET s.tenant_id = section.tenant_id,
                        s.knowledge_base_id = section.kb_id,
                        s.page_number = section.page_number,
                        s.chunk_type = section.chunk_type,
                        s.hierarchy_path = section.hierarchy_path

                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    doc_id=document_id,
                    sections=sections,
                )

            if children:
                await tx.run(
                    """
                    UNWIND $children AS child

                    MATCH (s:Section {chunk_id: child.parent_id})

                    MERGE (c:Chunk {chunk_id: child.chunk_id})
                    SET c.tenant_id = child.tenant_id,
                        c.knowledge_base_id = child.kb_id,
                        c.page_number = child.page_number,
                        c.chunk_type = child.chunk_type,
                        c.hierarchy_path = child.hierarchy_path

                    MERGE (s)-[:CONTAINS_CHUNK]->(c)
                    """,
                    children=children,
                )

        async with driver.session() as session:
            await session.execute_write(_write_tree)

        logger.info(
            "neo4j.document_tree_created",
            document_id=document_id,
            tenant_id=tenant_id,
            sections=len(parent_chunks),
            chunks=len(child_chunks),
        )

    async def add_supersedes_relationship(
        self, old_document_id: str, new_document_id: str
    ) -> None:
        """Link new document to old document via SUPERSEDES relationship."""
        driver = self._require_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (new_d:Document {id: $new_id})
                MATCH (old_d:Document {id: $old_id})
                MERGE (new_d)-[:SUPERSEDES]->(old_d)
                """,
                new_id=new_document_id,
                old_id=old_document_id,
            )

    async def delete_document(self, document_id: str, tenant_id: str = "default") -> None:
        """Detach delete document node and all associated section/chunk nodes."""
        driver = self._require_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[:CONTAINS_CHUNK]->(c:Chunk)
                DETACH DELETE d, s, c
                """,
                doc_id=document_id,
                tenant_id=tenant_id,
            )
        logger.info("neo4j.document_deleted", document_id=document_id, tenant_id=tenant_id)

    async def graph_search(
        self,
        entity_name: str,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Multi-hop relationship graph traversal query."""
        driver = self._require_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Chunk {tenant_id: $tenant_id, knowledge_base_id: $kb_id})
                WHERE c.hierarchy_path CONTAINS $entity OR c.chunk_id CONTAINS $entity
                RETURN c.chunk_id AS chunk_id, c.page_number AS page_number, c.hierarchy_path AS hierarchy_path
                LIMIT 20
                """,
                entity=entity_name,
                tenant_id=tenant_id,
                kb_id=knowledge_base_id,
            )
            records = await result.data()
            return records

    async def delete_document(self, document_id: str, tenant_id: str = "default") -> None:
        """Delete document node and all associated sections/chunks."""
        driver = self._require_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id})
                OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
                OPTIONAL MATCH (s)-[:CONTAINS_CHUNK]->(c:Chunk)
                DETACH DELETE d, s, c
                """,
                doc_id=document_id,
                tenant_id=tenant_id,
            )
        logger.info("neo4j.document_deleted", document_id=document_id, tenant_id=tenant_id)

    async def verify_connectivity(self) -> bool:
        """Return True if driver can reach Neo4j server."""
        try:
            if self._driver is None:
                await self.connect(retries=1)
            driver = self._require_driver()
            await driver.verify_connectivity()
            return True
        except Exception:
            return False

    async def health_check(self) -> dict[str, Any]:
        """Perform a live health check query against Neo4j."""
        start_time = time.perf_counter()
        try:
            if self._driver is None:
                await self.connect(retries=2, initial_delay=0.5)
            driver = self._require_driver()
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS ping")
                record = await result.single()
                if record and record["ping"] == 1:
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    return {"status": "ok", "latency_ms": latency_ms}
            return {"status": "error", "error": "Query returned unexpected result"}
        except Exception as err:
            return {"status": "error", "error": str(err)}

    def _require_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jClient.connect() must be called first")
        return self._driver

