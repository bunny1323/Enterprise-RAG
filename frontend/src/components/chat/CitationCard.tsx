import { FileText } from "lucide-react";
import type { Citation } from "../../types/chat";

interface CitationCardProps {
  citation: Citation;
  index: number;
  onOpen?: (citation: Citation) => void;
}

export function CitationCard({ citation, index, onOpen }: CitationCardProps) {
  return (
    <button
      onClick={() => onOpen?.(citation)}
      className="w-full text-left rounded-md border border-line bg-surface-sunk px-3 py-2.5 hover:border-signal transition-colors"
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium font-mono text-ink-soft">
          Source {index + 1}
        </span>
        {citation.weight !== undefined && (
          <span className="text-xs font-mono text-ink-faint">
            Relevance {citation.weight.toFixed(2)}
          </span>
        )}
      </div>
      <div className="flex items-center gap-1.5 text-sm text-ink font-medium">
        <FileText className="h-3.5 w-3.5 text-ink-faint shrink-0" />
        <span className="truncate">
          {citation.document_name || citation.document_id}
        </span>
      </div>
      {citation.page_number !== undefined && (
        <p className="text-xs text-ink-soft mt-0.5">
          Page {citation.page_number}
        </p>
      )}
      {citation.snippet && (
        <p className="text-xs text-ink-soft mt-1.5 line-clamp-2">
          &ldquo;{citation.snippet}&rdquo;
        </p>
      )}
    </button>
  );
}
