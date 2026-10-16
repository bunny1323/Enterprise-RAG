function bandFor(score: number): { label: string; className: string } {
  if (score >= 0.75) return { label: "High confidence", className: "text-good bg-good-soft" };
  if (score >= 0.45) return { label: "Medium confidence", className: "text-amber bg-amber-soft" };
  return { label: "Low confidence", className: "text-bad bg-bad-soft" };
}

export function ConfidenceBadge({ score }: { score?: number }) {
  if (score === undefined || score === null) return null;
  const { label, className } = bandFor(score);
  const pct = Math.round(score * 100);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium font-mono ${className}`}
    >
      {label} ({pct}%)
    </span>
  );
}
