import { FileText } from "lucide-react";
import type { RetrievalChunk } from "../../types/search";

export function RetrievalScore({ score }: { score?: number }) {
  if (score === undefined) return null;
  return (
    <span className="text-xs font-mono text-ink-soft">
      Score: {score.toFixed(2)}
    </span>
  );
}

export function RetrievalResult({ result, rank }: { result: RetrievalChunk; rank: number }) {
  return (
    <div className="rounded-md border border-line bg-surface px-4 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2 min-w-0">
          <span className="text-xs font-mono text-ink-faint mt-0.5">
            #{rank}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5 text-ink-faint shrink-0" />
              <span className="text-sm font-medium text-ink truncate">
                {result.document_name || result.document_id}
              </span>
            </div>
            {result.page_number !== undefined && (
              <p className="text-xs text-ink-soft mt-0.5">
                Page {result.page_number}
              </p>
            )}
            {result.content && (
              <p className="text-xs text-ink-soft mt-1.5 line-clamp-2">
                {result.content}
              </p>
            )}
          </div>
        </div>
        <RetrievalScore score={result.score} />
      </div>
    </div>
  );
}
