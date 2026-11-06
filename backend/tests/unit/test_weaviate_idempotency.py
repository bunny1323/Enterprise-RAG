from unittest.mock import MagicMock
from uuid import uuid4

from app.infrastructure.weaviate.client import WeaviateClient, _chunk_object_uuid
from app.models.chunk import Chunk


def test_weaviate_replay_uses_stable_chunk_uuid():
    chunk_id = f"{uuid4()}::p1::s0"
    assert _chunk_object_uuid(chunk_id) == _chunk_object_uuid(chunk_id)
    assert _chunk_object_uuid(chunk_id) != _chunk_object_uuid(f"{chunk_id}::child")


def test_batch_upsert_passes_deterministic_chunk_uuid():
    sdk_client = MagicMock()
    collection = MagicMock()
    batch = MagicMock()
    collection.batch.dynamic.return_value.__enter__.return_value = batch
    sdk_client.collections.get.return_value = collection

    weaviate = WeaviateClient("https://cluster.example", "test-key")
    weaviate._client = sdk_client
    chunk = Chunk(
        chunk_id="document::p1::s0",
        document_id=uuid4(),
        content="A manual chunk",
        page_number=1,
    )

    weaviate.upsert_chunks([chunk], [[0.1, 0.2]])

    batch.add_object.assert_called_once()
    assert batch.add_object.call_args.kwargs["uuid"] == _chunk_object_uuid(chunk.chunk_id)
