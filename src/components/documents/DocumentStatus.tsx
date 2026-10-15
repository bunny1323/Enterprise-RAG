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
  PENDING: {
    label: "Pending",
    className: "text-ink-soft bg-surface-wash",
    icon: Clock,
  },
  FAILED: {
    label: "Failed",
    className: "text-bad bg-bad-soft",
    icon: XCircle,
  },
};

export function DocumentStatusPill({ status }: { status: string }) {
  const config = CONFIG[status] ?? CONFIG.PENDING;
  const Icon = config.icon;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${config.className}`}
    >
      <Icon
        className={`h-3 w-3 ${status === "PROCESSING" ? "animate-spin" : ""}`}
      />
      {config.label}
    </span>
  );
}
