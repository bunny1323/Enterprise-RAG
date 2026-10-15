import { FileText } from "lucide-react";
import type { DocumentRecord } from "../../types/document";
import { DocumentStatusPill } from "../documents/DocumentStatus";
import { EmptyState } from "../common/EmptyState";

export function RecentDocuments({ documents }: { documents: DocumentRecord[] }) {
  if (documents.length === 0) {
    return (
      <div className="rounded-lg border border-line bg-surface">
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Upload your first enterprise document to start asking questions."
        />
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-line bg-surface divide-y divide-line">
      {documents.slice(0, 5).map((doc) => (
        <div
          key={doc.document_id}
          className="flex items-center justify-between px-5 py-3"
        >
          <div className="flex items-center gap-3 min-w-0">
            <FileText className="h-4 w-4 text-ink-faint shrink-0" />
            <span className="truncate text-sm text-ink">
              {doc.title || doc.file_name || doc.document_id}
            </span>
          </div>
          <DocumentStatusPill status={doc.status} />
        </div>
      ))}
    </div>
  );
}
