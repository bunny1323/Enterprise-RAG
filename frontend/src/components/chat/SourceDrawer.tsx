import { X, FileText } from "lucide-react";
import type { Citation } from "../../types/chat";

interface SourceDrawerProps {
  citation: Citation | null;
  onClose: () => void;
}

export function SourceDrawer({ citation, onClose }: SourceDrawerProps) {
  if (!citation) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-ink/30" onClick={onClose} />
      <div className="relative w-full max-w-sm bg-surface border-l border-line h-full overflow-y-auto shadow-xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-line">
          <h2 className="font-semibold text-ink text-sm">Source</h2>
          <button onClick={onClose} className="text-ink-faint hover:text-ink">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5 space-y-5">
          <div className="flex items-start gap-2">
            <FileText className="h-4 w-4 text-ink-faint mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-ink text-sm">
                {citation.document_name || citation.document_id}
              </p>
              {citation.page_number !== undefined && (
                <p className="text-xs text-ink-soft mt-0.5">
                  Page {citation.page_number}
                </p>
              )}
              {citation.chunk_id && (
                <p className="text-xs text-ink-faint font-mono mt-0.5">
                  Chunk {citation.chunk_id.slice(0, 12)}
                </p>
              )}
            </div>
          </div>

          <div>
            <p className="text-xs font-medium text-ink-soft mb-1.5">
              Relevant content
            </p>
            <div className="rounded-md bg-surface-sunk border border-line px-3 py-2.5 text-sm text-ink">
              &ldquo;{citation.snippet}&rdquo;
            </div>
          </div>

          {citation.weight !== undefined && (
            <div>
              <p className="text-xs font-medium text-ink-soft mb-1.5">
                Retrieval information
              </p>
              <div className="flex items-center justify-between rounded-md border border-line px-3 py-2 text-sm">
                <span className="text-ink-soft">Relevance score</span>
                <span className="font-mono text-ink">
                  {citation.weight.toFixed(2)}
                </span>
              </div>
            </div>
          )}

          <button
            onClick={onClose}
            className="w-full rounded-md border border-line-strong px-4 py-2 text-sm font-medium text-ink hover:bg-surface-wash transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
