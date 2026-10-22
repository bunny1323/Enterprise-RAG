import { useState, useEffect } from "react";
import { AppLayout } from "../components/layout/AppLayout";
import { API_BASE_URL } from "../services/api";
import { useHealth } from "../hooks/useHealth";
import { CheckCircle2, ExternalLink, Key, Lock, XCircle } from "lucide-react";

export default function Settings() {
  const { health, loading, checkHealth } = useHealth(false);
  const [token, setToken] = useState("");
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    const existing = sessionStorage.getItem("erag_token") || "";
    setToken(existing);
  }, []);

  const handleSaveToken = () => {
    if (token.trim()) {
      sessionStorage.setItem("erag_token", token.trim());
    } else {
      sessionStorage.removeItem("erag_token");
    }
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  const handleClearToken = () => {
    sessionStorage.removeItem("erag_token");
    setToken("");
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  return (
    <AppLayout title="Settings">
      <div className="max-w-xl mx-auto space-y-6">
        {/* Connection Settings */}
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

        {/* Authentication Token */}
        <div className="rounded-lg border border-line bg-surface px-5 py-4 space-y-3">
          <div className="flex items-center gap-2">
            <Key className="h-4 w-4 text-signal-dark" />
            <p className="text-sm font-medium text-ink">Session Authentication</p>
          </div>
          <p className="text-xs text-ink-soft">
            When API authentication is enabled on the backend, attach your Bearer token here. It is saved in session storage and transmitted securely over HTTPS/local connections.
          </p>
          <div className="space-y-2">
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-faint" />
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Enter API or Admin Token (Bearer secret)…"
                className="w-full rounded-md border border-line-strong bg-surface pl-9 pr-3 py-2 text-sm text-ink focus:border-signal"
              />
            </div>
            <div className="flex items-center gap-2 justify-end">
              {savedSuccess && (
                <span className="text-xs text-good font-medium mr-auto">Settings saved!</span>
              )}
              {token && (
                <button
                  type="button"
                  onClick={handleClearToken}
                  className="rounded px-3 py-1.5 text-xs text-bad hover:bg-bad/10 transition-colors"
                >
                  Clear Token
                </button>
              )}
              <button
                type="button"
                onClick={handleSaveToken}
                className="rounded bg-signal px-3.5 py-1.5 text-xs font-medium text-surface hover:bg-signal-dark transition-colors"
              >
                Save Token
              </button>
            </div>
          </div>
        </div>

        {/* Developer Links */}
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
