import { AppLayout } from "../components/layout/AppLayout";
import { API_BASE_URL } from "../services/api";
import { useHealth } from "../hooks/useHealth";
import { CheckCircle2, ExternalLink, XCircle } from "lucide-react";

export default function Settings() {
  const { health, loading, checkHealth } = useHealth(false);

  return (
    <AppLayout title="Settings">
      <div className="max-w-xl mx-auto space-y-6">
        <div className="rounded-lg border border-line bg-surface px-5 py-4">
          <p className="text-sm font-medium text-ink mb-3">Connection</p>
          <div className="flex items-center justify-between text-sm">
            <span className="text-ink-soft">Backend URL</span>
            <span className="font-mono text-ink">
              {API_BASE_URL || (import.meta.env.DEV ? "http://localhost:8001 (Vite Proxy)" : "Not configured")}
            </span>
          </div>
          <div className="mt-4 flex items-center justify-between rounded-md border border-line bg-surface-sunk px-3 py-2.5">
            <div className="flex items-center gap-2 text-sm">
              {health ? <CheckCircle2 className="h-4 w-4 text-good" /> : <XCircle className="h-4 w-4 text-bad" />}
              <span className="text-ink">{loading ? "Checking backend…" : health ? `Backend ${health.status}` : "Backend unavailable"}</span>
            </div>
            <button onClick={checkHealth} className="text-xs font-medium text-signal-dark hover:underline">Refresh</button>
          </div>
          <p className="text-xs text-ink-faint mt-2">
            Set via <code className="font-mono">VITE_API_BASE_URL</code> in
            your <code className="font-mono">.env</code> file.
          </p>
        </div>

        <div className="rounded-lg border border-line bg-surface px-5 py-4">
          <p className="text-sm font-medium text-ink mb-2">Developer links</p>
          <a href={`${API_BASE_URL || "http://localhost:8001"}/docs`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-sm text-signal-dark hover:underline">
            Open FastAPI documentation <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </AppLayout>
  );
}
