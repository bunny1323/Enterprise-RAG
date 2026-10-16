import { Loader2 } from "lucide-react";

export function Loading({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 justify-center text-ink-soft text-sm">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>{label ?? "Loading…"}</span>
    </div>
  );
}
