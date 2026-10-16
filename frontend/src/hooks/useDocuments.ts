import { useCallback, useEffect, useRef, useState } from "react";
import {
  getDocuments,
  getDocumentStatus,
  uploadDocument,
} from "../services/documents";
import type { DocumentRecord, DocumentStatus } from "../types/document";

const TERMINAL_STATUSES = new Set([
  "COMPLETED",
  "FAILED",
  "TIMEOUT",
  "CANCELLED",
  "QUARANTINED",
  "DUPLICATE",
]);
const POLL_INTERVAL_MS = 3000;

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>(
    {}
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      Object.values(pollTimers.current).forEach(clearTimeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollStatus = useCallback(
    (documentId: string, onUpdate?: (status: DocumentStatus) => void) => {
      const poll = async () => {
        try {
          const status = await getDocumentStatus(documentId);
          onUpdate?.(status);
          setDocuments((prev) =>
            prev.map((d) =>
              d.document_id === documentId
                ? {
                    ...d,
                    status: status.status,
                    indexed_chunks: status.indexed_chunks,
                    indexed_images: status.indexed_images,
                  }
                : d
            )
          );
          if (!TERMINAL_STATUSES.has(status.status)) {
            pollTimers.current[documentId] = setTimeout(
              poll,
              POLL_INTERVAL_MS
            );
          } else {
            delete pollTimers.current[documentId];
            refresh();
          }
        } catch {
          // stop polling on repeated failure; surfaced via document row error state
          delete pollTimers.current[documentId];
        }
      };
      poll();
    },
    [refresh]
  );

  const upload = useCallback(
    async (
      file: File,
      industry: string,
      title?: string,
      documentType?: string
    ) => {
      const res = await uploadDocument(file, industry, title, documentType);
      const optimistic: DocumentRecord = {
        document_id: res.document_id,
        title: title || file.name,
        file_name: file.name,
        industry,
        document_type: documentType,
        status: res.status || "PENDING",
      };
      setDocuments((prev) => [optimistic, ...prev]);
      pollStatus(res.document_id);
      return res;
    },
    [pollStatus]
  );

  return { documents, loading, error, refresh, upload, pollStatus };
}
