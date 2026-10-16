import { useHealth } from "../../hooks/useHealth";

const STATUS_STYLES: Record<string, { dot: string; label: string }> = {
  healthy: { dot: "bg-good", label: "System OK" },
  degraded: { dot: "bg-amber", label: "Degraded" },
  unhealthy: { dot: "bg-bad", label: "Unavailable" },
};

export function Topbar({ title }: { title: string }) {
  const { health, error } = useHealth(true);

  const statusKey = error
    ? "unhealthy"
    : health?.status && STATUS_STYLES[health.status]
    ? health.status
    : health
    ? "healthy"
    : "unhealthy";

  const style = STATUS_STYLES[statusKey] ?? STATUS_STYLES.unhealthy;

  return (
    <header className="flex items-center justify-between border-b border-line bg-surface px-6 py-4">
      <h1 className="text-lg font-semibold text-ink tracking-tight">
        {title}
      </h1>
      <div className="flex items-center gap-2 text-sm text-ink-soft">
        <span className={`h-2 w-2 rounded-full ${style.dot}`} />
        {style.label}
      </div>
    </header>
  );
}
