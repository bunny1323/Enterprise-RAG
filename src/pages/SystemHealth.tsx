import { AppLayout } from "../components/layout/AppLayout";
import { Loading } from "../components/common/Loading";
import { ErrorState } from "../components/common/ErrorState";
import { useHealth } from "../hooks/useHealth";

const STATUS_DOT: Record<string, string> = {
  connected: "bg-good",
  healthy: "bg-good",
  degraded: "bg-amber",
  disconnected: "bg-bad",
  unhealthy: "bg-bad",
};

export default function SystemHealth() {
  const { health, loading, error, checkHealth } = useHealth(true);

  return (
    <AppLayout title="System Health">
      <div className="max-w-2xl mx-auto space-y-6">
        {loading ? (
          <Loading label="Checking system…" />
        ) : error || !health ? (
          <ErrorState
            title="Backend unavailable"
            message={
              error ??
              "Please check whether the Enterprise-RAG backend is running."
            }
            onRetry={checkHealth}
          />
        ) : (
          <>
            <div className="rounded-lg border border-line bg-surface px-5 py-4 flex items-center justify-between">
              <div>
                <p className="text-sm text-ink-soft">Overall status</p>
                <p className="text-lg font-semibold text-ink capitalize mt-0.5">
                  {health.status}
                </p>
              </div>
              <span
                className={`h-3 w-3 rounded-full ${
                  STATUS_DOT[health.status] ?? "bg-ink-faint"
                }`}
              />
            </div>

            <div>
              <p className="text-sm font-medium text-ink mb-3">Services</p>
              <div className="rounded-lg border border-line bg-surface divide-y divide-line">
                {Object.entries(health.services).map(([name, status]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between px-5 py-3"
                  >
                    <span className="text-sm text-ink capitalize">
                      {name}
                    </span>
                    <span className="flex items-center gap-1.5 text-xs font-medium text-ink-soft capitalize">
                      <span
                        className={`h-2 w-2 rounded-full ${
                          STATUS_DOT[status] ?? "bg-ink-faint"
                        }`}
                      />
                      {status}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {health.version && (
              <div className="flex items-center justify-between text-sm text-ink-soft px-1">
                <span>Version</span>
                <span className="font-mono text-ink">{health.version}</span>
              </div>
            )}
          </>
        )}
      </div>
    </AppLayout>
  );
}
