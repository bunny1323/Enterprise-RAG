import { useState } from "react";
import { Plus, Search } from "lucide-react";
import { AppLayout } from "../components/layout/AppLayout";
import { DocumentTable } from "../components/documents/DocumentTable";
import { UploadModal } from "../components/documents/UploadModal";
import { ProcessingProgress } from "../components/documents/ProcessingProgress";
import { Loading } from "../components/common/Loading";
import { ErrorState } from "../components/common/ErrorState";
import { Modal } from "../components/common/Modal";
import { useDocuments } from "../hooks/useDocuments";
import type { DocumentRecord } from "../types/document";
import { getDocumentStatus } from "../services/documents";
import type { DocumentStatus } from "../types/document";

export default function Documents() {
  const { documents, loading, error, refresh, upload, pollStatus } =
    useDocuments();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<DocumentRecord | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<DocumentStatus | null>(
    null
  );

  const filtered = documents.filter((d) =>
    (d.title || d.file_name || "").toLowerCase().includes(query.toLowerCase())
  );

  const openDocument = async (doc: DocumentRecord) => {
    setSelected(doc);
    setSelectedStatus(null);
    try {
      const status = await getDocumentStatus(doc.document_id);
      setSelectedStatus(status);
      if (status.status === "PENDING" || status.status === "PROCESSING") {
        pollStatus(doc.document_id, setSelectedStatus);
      }
    } catch {
      // leave status panel showing what we already know from the list
    }
  };

  return (
    <AppLayout title="Documents">
      <div className="max-w-5xl mx-auto space-y-4">
        <div className="flex items-center justify-between">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-faint" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search documents…"
              className="w-full rounded-md border border-line-strong bg-surface pl-9 pr-3 py-2 text-sm text-ink focus:border-signal"
            />
          </div>
          <button
            onClick={() => setUploadOpen(true)}
            className="flex items-center gap-1.5 rounded-md bg-signal px-4 py-2 text-sm font-medium text-surface hover:bg-signal-dark transition-colors"
          >
            <Plus className="h-4 w-4" />
            Upload document
          </button>
        </div>

        <div className="rounded-lg border border-line bg-surface px-5 py-2 overflow-x-auto">
          {loading ? (
            <Loading label="Loading documents…" />
          ) : error ? (
            <ErrorState message={error} onRetry={refresh} />
          ) : (
            <DocumentTable documents={filtered} onSelect={openDocument} />
          )}
        </div>
      </div>

      <UploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUpload={upload}
      />

      <Modal
        open={!!selected}
        onClose={() => setSelected(null)}
        title={selected?.title || selected?.file_name || "Document"}
      >
        {selected && (
          <div className="space-y-4">
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-ink-soft">Industry</dt>
                <dd className="text-ink capitalize">
                  {selected.industry ?? "—"}
                </dd>
              </div>
              <div>
                <dt className="text-ink-soft">Type</dt>
                <dd className="text-ink">{selected.document_type ?? "—"}</dd>
              </div>
            </dl>
            <div>
              <p className="text-xs font-medium text-ink-soft mb-2">
                Processing status
              </p>
              {selectedStatus ? (
                <ProcessingProgress status={selectedStatus} />
              ) : (
                <Loading label="Checking status…" />
              )}
            </div>
          </div>
        )}
      </Modal>
    </AppLayout>
  );
}
