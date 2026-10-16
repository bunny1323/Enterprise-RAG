import { CheckCircle2, Loader2, XCircle, Clock } from "lucide-react";

const CONFIG: Record<
  string,
  { label: string; className: string; icon: typeof CheckCircle2 }
> = {
  COMPLETED: {
    label: "Ready",
    className: "text-good bg-good-soft",
    icon: CheckCircle2,
  },
  PROCESSING: {
    label: "Processing",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  PARSING: {
    label: "Parsing PDF",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  OCR: {
    label: "OCR Processing",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  VISION: {
    label: "Vision Analysis",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  CHUNKING: {
    label: "Chunking",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  METADATA: {
    label: "Enriching Metadata",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  EMBEDDING: {
    label: "Embedding",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  INDEXING: {
    label: "Indexing (Vector/Graph)",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  PENDING: {
    label: "Pending",
    className: "text-ink-soft bg-surface-wash",
    icon: Clock,
  },
  RECEIVED: {
    label: "Received",
    className: "text-ink-soft bg-surface-wash",
    icon: Clock,
  },
  VALIDATING: {
    label: "Validating",
    className: "text-amber bg-amber-soft",
    icon: Loader2,
  },
  FAILED: {
    label: "Failed",
    className: "text-bad bg-bad-soft",
    icon: XCircle,
  },
  TIMEOUT: {
    label: "Timed Out",
    className: "text-bad bg-bad-soft",
    icon: XCircle,
  },
  CANCELLED: {
    label: "Cancelled",
    className: "text-ink-soft bg-surface-wash",
    icon: XCircle,
  },
};

const IN_PROGRESS_STAGES = new Set([
  "PROCESSING",
  "PARSING",
  "OCR",
  "VISION",
  "CHUNKING",
  "METADATA",
  "EMBEDDING",
  "INDEXING",
  "VALIDATING",
]);

export function DocumentStatusPill({ status }: { status: string }) {
  const normStatus = (status || "").toUpperCase();
  const config = CONFIG[normStatus] ?? (IN_PROGRESS_STAGES.has(normStatus) ? CONFIG.PROCESSING : CONFIG.PENDING);
  const Icon = config.icon;
  const isSpinning = IN_PROGRESS_STAGES.has(normStatus);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${config.className}`}
    >
      <Icon
        className={`h-3 w-3 ${isSpinning ? "animate-spin" : ""}`}
      />
      {config.label}
    </span>
  );
}
