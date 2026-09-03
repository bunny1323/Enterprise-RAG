"""
Document API routes.
POST /api/v1/documents      — upload PDF, get document_id in <100ms
GET  /api/v1/documents/{id}/status — poll ingestion status
"""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from app.api.dependencies import PostgresDep, SupervisorDep, TenantContextDep, get_supervisor
from app.agents.supervisor.agent import IngestionSupervisor
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
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
