import asyncio
import os
import sys

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config.settings import get_settings
from app.infrastructure.weaviate.client import WeaviateClient
from app.infrastructure.postgres.client import PostgresClient
from app.services.embeddings.local_provider import LocalEmbeddingProvider
from app.config.logging import configure_logging, get_logger

configure_logging(debug=True)
logger = get_logger(__name__)

async def migrate():
    settings = get_settings()
    
    # 1. Initialize dependencies
    logger.info("Connecting to Weaviate...")
    weaviate_client = WeaviateClient(url=settings.weaviate_url, api_key=settings.weaviate_api_key)
    weaviate_client.connect()
    
    logger.info("Connecting to Postgres...")
    postgres = PostgresClient(database_url=settings.database_url)
    await postgres.init_pool()

    embedder = LocalEmbeddingProvider(
        model_name=settings.local_embedding_model,
        device=settings.local_embedding_device,
        batch_size=settings.local_embedding_batch_size,
    )
    
    try:
        # 1. Detect existing dimension
        existing_dim = weaviate_client.get_collection_dimension()
        object_count = weaviate_client.get_object_count()
        target_dim = embedder.get_dimension()
        
        logger.info(f"Existing Weaviate Dimension: {existing_dim}")
        logger.info(f"Target Embedder Dimension: {target_dim}")
        logger.info(f"Current Object Count: {object_count}")
        
        if existing_dim == target_dim:
            logger.info("Dimensions match. No migration required.")
            return

        # 2. Report incompatibility
        if existing_dim is not None:
            logger.warning("==================================================")
            logger.warning(f"INCOMPATIBLE VECTOR SPACE DETECTED (Found {existing_dim}, expected {target_dim})")
            logger.warning("==================================================")
            
            confirm = input("Type 'CONFIRM' to delete the collection and re-index all chunks: ")
            if confirm.strip() != "CONFIRM":
                logger.error("Migration aborted by user.")
                return

        # 3. Create/use a BGE-compatible collection
        logger.info("Recreating Weaviate collection...")
        weaviate_client.recreate_collection()
        
        # 4 & 5. Preserve metadata and re-index
        # Fetch all chunks from Postgres
        logger.info("Fetching chunks from PostgreSQL...")
        chunks_records = await postgres.fetch("SELECT * FROM chunks")
        logger.info(f"Found {len(chunks_records)} chunks in Postgres to migrate.")
        
        from app.models.chunk import Chunk, ChunkType
        chunks = []
        for r in chunks_records:
            metadata = r.get("metadata") or {}
            bbox = r.get("bounding_box")
            import json
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if isinstance(bbox, str):
                bbox = json.loads(bbox)
            
            chunk = Chunk(
                chunk_id=r["chunk_id"],
                parent_id=r.get("parent_id"),
                document_id=r["document_id"],
                tenant_id=r["tenant_id"],
                assistant_id=r["assistant_id"],
                knowledge_base_id=r["knowledge_base_id"],
                content=r["content"],
                content_hash=r["content_hash"],
                section=r.get("section"),
                subsection=r.get("subsection"),
                context_prefix=r.get("context_prefix"),
                embedding_representation=r.get("embedding_representation") or "text",
                page_number=r["page_number"],
                bounding_box=bbox,
                chunk_type=ChunkType(r["chunk_type"]) if r.get("chunk_type") else ChunkType.TEXT,
                access_classification=r["access_classification"],
                industry_domain=r["industry_domain"],
                hierarchy_path=r.get("hierarchy_path"),
                metadata=metadata,
            )
            chunks.append(chunk)

        if not chunks:
            logger.info("No chunks in Postgres to migrate.")
            return
            
        logger.info("Generating embeddings for all chunks...")
        texts = [c.content for c in chunks]
        vectors = embedder.embed_batch(texts)
        
        logger.info("Upserting vectors to Weaviate...")
        # Since we just do a bulk reindex, doing it in one batch is okay for local dev sizes.
        # For production, we would chunk the lists.
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            weaviate_client.upsert_chunks(
                chunks=chunks[i:i+batch_size],
                vectors=vectors[i:i+batch_size]
            )
            logger.info(f"Upserted {min(i+batch_size, len(chunks))}/{len(chunks)}...")

        # Update cache as well to clear out old embeddings? We should clear the embedding cache.
        logger.info("Clearing PostgreSQL embedding_cache to remove old dim=1024 vectors...")
        await postgres.execute("TRUNCATE embedding_cache")
            
        # 6. Verify vector dimension
        new_dim = weaviate_client.get_collection_dimension()
        logger.info(f"Verification: New Dimension = {new_dim}")
        
        # 7. Verify object count
        new_count = weaviate_client.get_object_count()
        logger.info(f"Verification: New Object Count = {new_count}")
        
        if new_dim != target_dim:
            logger.error(f"MIGRATION FAILED: Expected dim {target_dim}, got {new_dim}")
        else:
            logger.info("MIGRATION COMPLETED SUCCESSFULLY.")
            
    finally:
        weaviate_client.close()
        await postgres.close()

if __name__ == "__main__":
    asyncio.run(migrate())
