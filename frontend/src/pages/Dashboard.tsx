import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle2,
  Loader2,
  XCircle,
} from "lucide-react";

import { AppLayout } from "../components/layout/AppLayout";
import { StatCard } from "../components/dashboard/StatCard";
import { RecentDocuments } from "../components/dashboard/RecentDocuments";
import { Loading } from "../components/common/Loading";
import { ErrorState } from "../components/common/ErrorState";
import { useDocuments } from "../hooks/useDocuments";

export default function Dashboard() {
  const {
    documents: rawDocuments,
    loading,
    error,
    refresh,
  } = useDocuments();

  const [queryCount, setQueryCount] = useState<number | null>(null);
  const navigate = useNavigate();

  // Normalize the response so the dashboard always receives an array.
  const documents = Array.isArray(rawDocuments)
    ? rawDocuments
    : Array.isArray(
          (rawDocuments as unknown as { documents?: unknown[] })?.documents
        )
      ? (
          rawDocuments as unknown as {
            documents: typeof rawDocuments;
          }
        ).documents
      : [];

  useEffect(() => {
    // Query count remains unavailable until a documented endpoint exists.
    setQueryCount(null);
  }, []);

  const totalChunks = documents.reduce(
    (sum, document) => sum + (document.indexed_chunks ?? 0),
    0
  );

  const ready = documents.filter(
    (document) => document.status === "COMPLETED"
  ).length;

  const processing = documents.filter(
    (document) =>
      document.status === "PROCESSING" ||
      document.status === "PENDING"
  ).length;

  const failed = documents.filter(
    (document) => document.status === "FAILED"
  ).length;

  return (
    <AppLayout title="Dashboard">
      <div className="mx-auto max-w-5xl space-y-8">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-ink">
            Enterprise Document Intelligence
          </h2>

          <p className="mt-1 max-w-lg text-sm text-ink-soft">
            Ask questions, retrieve knowledge and get grounded answers
            from your enterprise documents.
          </p>
        </div>

        {loading ? (
          <Loading label="Loading dashboard…" />
        ) : error ? (
          <ErrorState message={error} onRetry={refresh} />
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard
                label="Documents"
                value={documents.length}
              />

              <StatCard
                label="Indexed chunks"
                value={totalChunks > 0 ? totalChunks : null}
              />

              <StatCard
                label="Queries"
                value={queryCount}
              />

              <div className="col-span-2 rounded-lg border border-line bg-surface px-5 py-4 shadow-card lg:col-span-4">
                <div className="flex flex-wrap items-center gap-5 text-sm">
                  <span className="font-medium text-ink">
                    Document pipeline
                  </span>

                  <span className="inline-flex items-center gap-1.5 text-good">
                    <CheckCircle2 className="h-4 w-4" />
                    {ready} ready
                  </span>

                  <span className="inline-flex items-center gap-1.5 text-amber">
                    <Loader2 className="h-4 w-4" />
                    {processing} processing
                  </span>

                  <span className="inline-flex items-center gap-1.5 text-bad">
                    <XCircle className="h-4 w-4" />
                    {failed} failed
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-sm font-medium text-ink">
                Recent documents
              </h3>

              <RecentDocuments documents={documents} />
            </div>

            <div className="flex justify-center pt-2">
              <button
                type="button"
                onClick={() => navigate("/chat")}
                className="rounded-md bg-signal px-5 py-2.5 text-sm font-medium text-surface transition-colors hover:bg-signal-dark"
              >
                Ask Enterprise AI
              </button>
            </div>
          </>
        )}
      </div>
    </AppLayout>
  );
}
