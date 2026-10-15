interface StatCardProps {
  label: string;
  value: string | number | null | undefined;
}

export function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="rounded-lg border border-line bg-surface px-5 py-4 shadow-card">
      <p className="text-sm text-ink-soft">{label}</p>
      <p className="mt-1 text-2xl font-semibold font-mono text-ink">
        {value === null || value === undefined ? "—" : value}
      </p>
    </div>
  );
}
