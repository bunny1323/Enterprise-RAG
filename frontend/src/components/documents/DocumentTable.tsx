import { FileText, MoreVertical } from "lucide-react";
import type { DocumentRecord } from "../../types/document";
import { DocumentStatusPill } from "./DocumentStatus";
import { EmptyState } from "../common/EmptyState";

interface DocumentTableProps {
  documents: DocumentRecord[];
  onSelect?: (doc: DocumentRecord) => void;
}

export function DocumentTable({ documents, onSelect }: DocumentTableProps) {
  if (documents.length === 0) {
    return (
      <EmptyState
        icon={FileText}
        title="No documents found"
        description="Upload your first document to start using Enterprise-RAG."
      />
    );
  }

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-line text-left text-ink-soft">
          <th className="py-2.5 font-medium">Name</th>
          <th className="py-2.5 font-medium">Industry</th>
          <th className="py-2.5 font-medium">Status</th>
          <th className="py-2.5 font-medium w-8"></th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line">
        {documents.map((doc) => (
          <tr
            key={doc.document_id}
            className="cursor-pointer hover:bg-surface-wash/60 transition-colors"
            onClick={() => onSelect?.(doc)}
          >
            <td className="py-3">
              <div className="flex items-center gap-2.5 min-w-0">
                <FileText className="h-4 w-4 text-ink-faint shrink-0" />
                <span className="truncate text-ink">
                  {doc.title || doc.file_name || doc.document_id}
                </span>
              </div>
            </td>
            <td className="py-3 text-ink-soft capitalize">
              {doc.industry ?? "—"}
            </td>
            <td className="py-3">
              <DocumentStatusPill status={doc.status} />
            </td>
            <td className="py-3 text-right">
              <button
                aria-label="More actions"
                onClick={(e) => e.stopPropagation()}
                className="text-ink-faint hover:text-ink"
              >
                <MoreVertical className="h-4 w-4" />
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
