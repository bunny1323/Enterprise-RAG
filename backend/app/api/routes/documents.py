"""
Document API routes.
POST /api/v1/documents      — upload PDF, get document_id in <100ms
GET  /api/v1/documents/{id}/status — poll ingestion status
"""
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status

from app.api.dependencies import PostgresDep, SupervisorDep, TenantContextDep, get_supervisor
from app.agents.supervisor.agent import IngestionSupervisor
from app.config.logging import get_logger
from app.models.document import DocumentStatusResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF document for ingestion",
    response_description="Document ID and initial PENDING status",
)
async def upload_document(
    file: UploadFile,
    industry: str = Form(default="manufacturing"),
    tenant_ctx: TenantContextDep = ...,  # type: ignore[assignment]
    supervisor: Annotated[IngestionSupervisor, Depends(get_supervisor)] = ...,  # type: ignore[assignment]
) -> dict:
    """
    Accept a PDF upload and queue it for asynchronous ingestion.

    Returns immediately with document_id in <100ms.
    Poll GET /api/v1/documents/{id}/status for progress.

    - **file**: Multipart PDF file (max 100 MB)
    - **industry**: Industry domain for metadata enrichment (default: manufacturing)
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    if not file.content_type or "pdf" not in file.content_type.lower():
        # Pre-flight MIME check — full validation happens in step 01
        logger.warning(
            "upload.suspicious_content_type",
            content_type=file.content_type,
            filename=file.filename,
        )

    try:
        result = await supervisor.handle_upload(
            file=file,
            industry=industry,
            tenant_id=tenant_ctx.tenant_id,
            assistant_id=tenant_ctx.assistant_id,
            knowledge_base_id=tenant_ctx.knowledge_base_id,
        )
        logger.info(
            "upload.accepted",
            document_id=result["document_id"],
            filename=file.filename,
            industry=industry,
        )
        return result
    except Exception as err:
        logger.error("upload.failed", error=str(err), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(err)}",
        ) from err


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document ingestion status",
)
async def get_document_status(
    document_id: str,
    postgres: PostgresDep,
    tenant_ctx: TenantContextDep,
) -> DocumentStatusResponse:
    """
    Poll the ingestion status of a document.

    Returns current status, progress percentage, and timestamps.
    """
    # Validate UUID format
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document_id format: {document_id}",
        )

    row = await postgres.fetchrow(
        """
        SELECT id, status, progress_percent, created_at, completed_at, error_message
        FROM documents
        WHERE id = $1 AND tenant_id = $2 AND knowledge_base_id = $3
        """,
        doc_uuid,
        tenant_ctx.tenant_id,
        tenant_ctx.knowledge_base_id,
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentStatusResponse(
        document_id=row["id"],
        status=row["status"],
        progress_percent=row["progress_percent"],
        created_at=row["created_at"],
        completed_at=row.get("completed_at"),
        error_message=row.get("error_message"),
    )


@router.get(
    "",
    summary="List documents with optional status filter",
)
async def list_documents(
    postgres: PostgresDep,
    tenant_ctx: TenantContextDep,
    status_filter: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """
    List ingested documents with optional status filtering.

    - **status_filter**: Filter by status (PENDING, COMPLETED, FAILED, etc.)
    - **limit**: Max results (default 20, max 100)
    - **offset**: Pagination offset
    """
    limit = min(limit, 100)

    if status_filter:
        rows = await postgres.fetch(
            """
            SELECT id, file_name, industry, status, progress_percent,
                   page_count, created_at, completed_at
            FROM documents
            WHERE status = $1 AND tenant_id = $2 AND knowledge_base_id = $3
            ORDER BY created_at DESC
            LIMIT $4 OFFSET $5
            """,
            status_filter.upper(),
            tenant_ctx.tenant_id,
            tenant_ctx.knowledge_base_id,
            limit,
            offset,
        )
    else:
        rows = await postgres.fetch(
            """
            SELECT id, file_name, industry, status, progress_percent,
                   page_count, created_at, completed_at
            FROM documents
            WHERE tenant_id = $1 AND knowledge_base_id = $2
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
            """,
            tenant_ctx.tenant_id,
            tenant_ctx.knowledge_base_id,
            limit,
            offset,
        )

    return {
        "documents": [
            {
                "document_id": str(r["id"]),
                "file_name": r["file_name"],
                "industry": r["industry"],
                "status": r["status"],
                "progress_percent": r["progress_percent"],
                "page_count": r["page_count"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            }
            for r in rows
        ],
        "count": len(rows),
        "limit": limit,
        "offset": offset,
    }


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete document and purge from all storage backends",
)
async def delete_document(
    document_id: str,
    postgres: PostgresDep,
    tenant_ctx: TenantContextDep,
    request: Request,
) -> dict[str, Any]:
    """
    Completely delete a document:
    1. Cascading delete from PostgreSQL (documents, chunks, jobs, checkpoints, document_structure, indexing_state)
    2. Purge from Weaviate vector database
    3. Purge from Neo4j knowledge graph
    4. Purge uploaded raw file and extracted figure images from disk
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document_id format: {document_id}",
        )

    # 1. Fetch document to verify existence and retrieve storage_path & tenant
    doc = await postgres.fetchrow(
        "SELECT id, tenant_id, knowledge_base_id, storage_path, file_name FROM documents WHERE id = $1",
        doc_uuid,
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Multi-tenant check: enforce only if caller provided a non-default tenant
    if tenant_ctx.tenant_id != "default" and doc["tenant_id"] != tenant_ctx.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: document belongs to a different tenant",
        )

    actual_tenant_id = doc["tenant_id"] or "default"

    # 2. Delete from Weaviate vector database
    try:
        weaviate_client = getattr(request.app.state, "weaviate", None)
        if weaviate_client:
            weaviate_client.delete_by_document(document_id=document_id, tenant_id=actual_tenant_id)
    except Exception as e:
        logger.warning("documents.weaviate_delete_warning", error=str(e), document_id=document_id)

    # 3. Delete from Neo4j knowledge graph
    try:
        neo4j_client = getattr(request.app.state, "neo4j", None)
        if neo4j_client:
            await neo4j_client.delete_document(document_id=document_id, tenant_id=actual_tenant_id)
    except Exception as e:
        logger.warning("documents.neo4j_delete_warning", error=str(e), document_id=document_id)

    # 4. Clean up disk files (raw uploaded file and extracted figure images)
    storage_path = doc.get("storage_path")
    if storage_path:
        try:
            import os
            if os.path.isfile(storage_path):
                os.remove(storage_path)
                logger.info("documents.storage_file_deleted", path=storage_path)
        except Exception as e:
            logger.warning("documents.storage_delete_warning", error=str(e), path=storage_path)

        try:
            from pathlib import Path
            import shutil

            settings = getattr(request.app.state, "settings", None)
            if settings:
                doc_stem = Path(storage_path).stem
                if doc_stem:
                    figures_dir = Path(settings.processed_storage_path) / "figures" / doc_stem
                    if figures_dir.exists() and figures_dir.is_dir():
                        shutil.rmtree(figures_dir, ignore_errors=True)
                        logger.info("documents.figures_directory_deleted", path=str(figures_dir))
        except Exception as e:
            logger.warning("documents.figures_delete_warning", error=str(e))

    # 5. Delete from PostgreSQL (safely handles FKs and all dependent tables)
    deleted = await postgres.delete_document(doc_uuid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} was not found or already deleted",
        )

    logger.info("documents.deleted_successfully", document_id=document_id, tenant=actual_tenant_id)

    return {
        "status": "deleted",
        "document_id": document_id,
        "detail": "Purged from PostgreSQL, Weaviate, Neo4j, and local storage.",
    }
