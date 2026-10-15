import type { DocumentStatus } from "../../types/document";

export function ProcessingProgress({ status }: { status: DocumentStatus }) {
  if (status.status === "FAILED") {
    return (
      <div className="text-sm text-bad">
        Document processing failed.
        {status.error && (
          <p className="text-xs text-ink-soft mt-1">{status.error}</p>
        )}
      </div>
    );
  }

  if (status.status === "COMPLETED") {
    return (
      <div className="text-sm text-ink-soft space-x-4 font-mono">
        {status.indexed_chunks !== undefined && (
          <span>Chunks: {status.indexed_chunks}</span>
        )}
        {status.indexed_images !== undefined && (
          <span>Images: {status.indexed_images}</span>
        )}
        {status.elapsed_seconds !== undefined && (
          <span>{status.elapsed_seconds}s</span>
        )}
      </div>
    );
  }

  const pct = status.progress_percent ?? undefined;

  return (
    <div className="space-y-1.5">
      {pct !== undefined ? (
        <div className="h-1.5 w-full rounded-full bg-surface-wash overflow-hidden">
          <div
            className="h-full bg-signal transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : (
        <div className="h-1.5 w-full rounded-full bg-surface-wash overflow-hidden">
          <div className="h-full w-1/3 bg-signal animate-pulse" />
        </div>
      )}
      <div className="flex gap-4 text-xs text-ink-soft font-mono">
        {status.indexed_chunks !== undefined && (
          <span>Chunks indexed: {status.indexed_chunks}</span>
        )}
        {status.indexed_images !== undefined && (
          <span>Images indexed: {status.indexed_images}</span>
        )}
      </div>
    </div>
  );
}
