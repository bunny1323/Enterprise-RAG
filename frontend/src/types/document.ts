export type DocumentStatusValue =
  | "PENDING"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED";

export interface UploadResponse {
  document_id: string;
  status: DocumentStatusValue | string;
  message?: string;
}

export interface DocumentStatus {
  document_id: string;
  status: DocumentStatusValue | string;
  progress_percent?: number;
  indexed_chunks?: number;
  indexed_images?: number;
  elapsed_seconds?: number;
  error?: string;
}

export interface DocumentRecord {
  document_id: string;
  title?: string;
  file_name?: string;
  industry?: string;
  document_type?: string;
  status: DocumentStatusValue | string;
  created_at?: string;
  indexed_chunks?: number;
  indexed_images?: number;
}

export const INDUSTRY_OPTIONS = [
  "manufacturing",
  "healthcare",
  "legal",
  "finance",
  "aerospace",
] as const;

export type Industry = (typeof INDUSTRY_OPTIONS)[number];
