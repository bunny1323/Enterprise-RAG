import { api, toApiError } from "./api";
import type {
  DocumentRecord,
  DocumentStatus,
  UploadResponse,
} from "../types/document";

/**
 * Endpoint note: the handoff docs disagree on whether this is
 * POST /api/v1/documents or POST /api/v1/documents/upload. The detailed
 * spec (and the running backend's /docs) is the source of truth — this
 * uses /api/v1/documents. Update here if the live schema differs.
 */
const DOCUMENTS_PATH = "/api/v1/documents";

export async function uploadDocument(
  file: File,
  industry: string,
  title?: string,
  documentType?: string
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("industry", industry || "manufacturing");
  if (title) formData.append("title", title);
  if (documentType) formData.append("document_type", documentType);

  try {
    const res = await api.post<UploadResponse>(DOCUMENTS_PATH, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function getDocumentStatus(
  documentId: string
): Promise<DocumentStatus> {
  try {
    const res = await api.get<any>(
      `${DOCUMENTS_PATH}/${documentId}/status`
    );
    const data = res.data;
    return {
      document_id: String(data.document_id || documentId),
      status: data.status,
      progress_percent: data.progress_percent,
      indexed_chunks: data.indexed_chunks,
      indexed_images: data.indexed_images,
      elapsed_seconds: data.elapsed_seconds,
      error: data.error || data.error_message,
    };
  } catch (err) {
    throw toApiError(err);
  }
}

export async function getDocuments(): Promise<DocumentRecord[]> {
  try {
    const res = await api.get<any>(DOCUMENTS_PATH);
    const rawList = Array.isArray(res.data)
      ? res.data
      : Array.isArray(res.data?.documents)
      ? res.data.documents
      : [];

    return rawList.map((doc: any) => ({
      document_id: String(doc.document_id || doc.id),
      title: doc.title || doc.file_name || doc.document_id,
      file_name: doc.file_name,
      industry: doc.industry,
      document_type: doc.document_type,
      status: doc.status || "PENDING",
      created_at: doc.created_at,
      indexed_chunks: doc.indexed_chunks,
      indexed_images: doc.indexed_images,
    }));
  } catch (err) {
    throw toApiError(err);
  }
}

export async function deleteDocument(documentId: string): Promise<{ status: string; document_id: string }> {
  try {
    const res = await api.delete<{ status: string; document_id: string }>(
      `${DOCUMENTS_PATH}/${documentId}`
    );
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}
