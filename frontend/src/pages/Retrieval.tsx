import { useState } from "react";
import { Search } from "lucide-react";
import { AppLayout } from "../components/layout/AppLayout";
import { RetrievalPipeline } from "../components/retrieval/RetrievalPipeline";
import { RetrievalResult } from "../components/retrieval/RetrievalResult";
import { Loading } from "../components/common/Loading";
import { ErrorState } from "../components/common/ErrorState";
import { EmptyState } from "../components/common/EmptyState";
import { runSearch } from "../services/search";
import type { RetrievalChunk } from "../types/search";
import { INDUSTRY_OPTIONS } from "../types/document";

export default function Retrieval() {
  const [query, setQuery] = useState("");
  const [industry, setIndustry] = useState("");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<RetrievalChunk[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runSearch({
        query,
        industry: industry || null,
        top_k: topK,
      });
      setResults(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppLayout title="Retrieval Explorer">
      <div className="max-w-3xl mx-auto space-y-6">
        <p className="text-sm text-ink-soft">
          Inspect hybrid retrieval results directly, without LLM generation.
        </p>

        <div className="rounded-lg border border-line bg-surface p-4 space-y-3">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-faint" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && search()}
                placeholder="hydraulic pressure"
                className="w-full rounded-md border border-line-strong bg-surface-sunk pl-9 pr-3 py-2 text-sm text-ink focus:border-signal"
              />
            </div>
            <button
              onClick={search}
              className="rounded-md bg-signal px-4 py-2 text-sm font-medium text-surface hover:bg-signal-dark transition-colors"
            >
              Search
            </button>
          </div>
          <div className="flex gap-4">
            <label className="flex items-center gap-1.5 text-xs text-ink-soft">
              Industry
              <select
                value={industry}
                onChange={(e) => setIndustry(e.target.value)}
                className="rounded-md border border-line-strong bg-surface-sunk px-2 py-1 text-xs text-ink"
              >
                <option value="">All industries</option>
                {INDUSTRY_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt.charAt(0).toUpperCase() + opt.slice(1)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-xs text-ink-soft">
              Top K
              <input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-16 rounded-md border border-line-strong bg-surface-sunk px-2 py-1 text-xs text-ink"
              />
            </label>
          </div>
        </div>

        <RetrievalPipeline />

        <div>
          <p className="text-sm font-medium text-ink mb-3">Results</p>
          {loading ? (
            <Loading label="Searching knowledge base…" />
          ) : error ? (
            <ErrorState message={error} onRetry={search} />
          ) : results === null ? (
            <EmptyState
              icon={Search}
              title="Enter a query to inspect retrieval results."
            />
          ) : results.length === 0 ? (
            <EmptyState
              title="No relevant information was found"
              description="No results matched this query in the available documents."
            />
          ) : (
            <div className="space-y-2">
              {results.map((r, i) => (
                <RetrievalResult key={r.chunk_id || i} result={r} rank={i + 1} />
              ))}
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
