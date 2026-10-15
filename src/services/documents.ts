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
  formData.append("industry", industry);
  if (title) formData.append("title", title);
  if (documentType) formData.append("document_type", documentType);

  try {
    const res = await api.post<UploadResponse>(DOCUMENTS_PATH, formData);
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function getDocumentStatus(
  documentId: string
): Promise<DocumentStatus> {
  try {
    const res = await api.get<DocumentStatus>(
      `${DOCUMENTS_PATH}/${documentId}/status`
    );
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function getDocuments(): Promise<DocumentRecord[]> {
  try {
    const res = await api.get<DocumentRecord[]>(DOCUMENTS_PATH);
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}
