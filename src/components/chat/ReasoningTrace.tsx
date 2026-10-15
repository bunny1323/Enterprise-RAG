import { Loader2 } from "lucide-react";

const STEP_LABELS: Record<string, string> = {
  cache_check: "Checking knowledge cache",
  retrieval: "Retrieving relevant information",
  rerank: "Reranking results",
  generation: "Generating answer",
};

export function ReasoningTrace({ steps, active }: { steps: string[]; active: boolean }) {
  if (steps.length === 0 && active) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-soft">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Generating answer…
      </div>
    );
  }

  if (steps.length === 0) return null;

  return (
    <div className="space-y-1">
      {steps.map((step, i) => (
        <div key={i} className="flex items-center gap-2 text-sm text-ink-soft">
          {active && i === steps.length - 1 ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
          ) : (
            <span className="h-1.5 w-1.5 rounded-full bg-signal shrink-0" />
          )}
          {STEP_LABELS[step] || step}
        </div>
      ))}
    </div>
  );
}
