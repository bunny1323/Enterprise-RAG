"""
Authenticated and tenant-isolated image serving endpoint.
Replaces public StaticFiles mount to enforce document-level authorization.
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.api.dependencies import PostgresDep, SettingsDep, TenantContextDep
from app.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/images", tags=["images"])

ALLOWED_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@router.get(
    "/{image_path:path}",
    summary="Retrieve an authenticated and authorized document image",
    response_description="Binary image file",
)
async def get_document_image(
    image_path: str,
    tenant_ctx: TenantContextDep,
    postgres: PostgresDep,
    settings: SettingsDep,
) -> FileResponse:
    """
    Serve document images with strict multi-tenant authorization and path traversal defenses.

    Security controls:
    1. Authentication is required via `TenantContextDep`.
    2. Extension whitelist restricts files to valid image types.
    3. Resolves path within `settings.processed_storage_path` using `is_relative_to`
       to eliminate directory traversal (`../`).
    4. Resolves document ownership via PostgreSQL storage path query.
    5. Returns HTTP 404 for missing files, invalid paths, AND unauthorized access
       to prevent enumeration/existence probing of other tenants' data.
    """
    ext = Path(image_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    base_dir = Path(settings.processed_storage_path).resolve()
    try:
        target_file = (base_dir / image_path).resolve()
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    # Prevent directory traversal
    if not target_file.is_relative_to(base_dir):
        logger.warning(
            "security.path_traversal_prevented",
            attempted_path=image_path,
            caller_tenant=tenant_ctx.tenant_id,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    if not target_file.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    # Resolve associated document from path structure:
    # Typical storage structure is: processed/figures/<doc_stem>/page<N>_<id>.png
    rel_parts = target_file.relative_to(base_dir).parts
    doc_stem = None
    if len(rel_parts) >= 2:
        doc_stem = rel_parts[1] if rel_parts[0] == "figures" else rel_parts[0]

    is_admin = tenant_ctx.claims.get("is_admin", False)

    if doc_stem:
        try:
            row = await postgres.fetchrow(
                "SELECT id, tenant_id FROM documents WHERE storage_path LIKE $1 LIMIT 1",
                f"%{doc_stem}%",
            )
            if row:
                doc_tenant = row.get("tenant_id") or "default"
                if doc_tenant != tenant_ctx.tenant_id and not is_admin:
                    logger.warning(
                        "security.unauthorized_image_access_prevented",
                        image=image_path,
                        doc_tenant=doc_tenant,
                        caller_tenant=tenant_ctx.tenant_id,
                    )
                    # Return 404 so caller cannot distinguish between non-existent files and other tenants' files
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
            else:
                # Document row not found: in production with auth enabled, reject access
                if not is_admin and (settings.is_production or settings.api_key_auth_enabled):
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
        except HTTPException:
            raise
        except Exception as err:
            logger.error("security.image_ownership_check_error", error=str(err))
            if not is_admin:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    return FileResponse(
        path=str(target_file),
        media_type=ALLOWED_EXTENSIONS[ext],
        headers={"Cache-Control": "private, max-age=3600"},
    )
