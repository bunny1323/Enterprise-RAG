import { useEffect, useState } from "react";
import { AppLayout } from "../components/layout/AppLayout";
import { StatCard } from "../components/dashboard/StatCard";
import { Loading } from "../components/common/Loading";
import { ErrorState } from "../components/common/ErrorState";
import { getEvaluation } from "../services/system";
import type { EvaluationMetrics } from "../types/system";

const METRIC_LABELS: Record<string, string> = {
  faithfulness: "Faithfulness",
  context_precision: "Context precision",
  context_recall: "Context recall",
  layout_extraction_accuracy: "Layout extraction accuracy",
};

export default function Evaluation() {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [forbidden, setForbidden] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    setForbidden(false);
    try {
      const res = await getEvaluation();
      setMetrics(res);
    } catch (err) {
      const status = (err as { status?: number }).status;
      if (status === 401 || status === 403) {
        setForbidden(true);
      } else {
        setError((err as Error).message);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // This endpoint requires an admin:read scope. If the current user can't
  // access it, hide the page's metrics rather than showing fabricated data.
  if (forbidden) {
    return (
      <AppLayout title="Evaluation">
        <div className="max-w-2xl mx-auto">
          <ErrorState
            title="Not available"
            message="Evaluation metrics require admin access, which this session doesn't have."
          />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout title="Evaluation">
      <div className="max-w-3xl mx-auto space-y-6">
        <p className="text-sm text-ink-soft">
          RAGAS metrics reported by the backend's evaluation pipeline.
        </p>

        {loading ? (
          <Loading label="Loading evaluation metrics…" />
        ) : error ? (
          <ErrorState message={error} onRetry={load} />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {Object.entries(METRIC_LABELS).map(([key, label]) => (
              <StatCard
                key={key}
                label={label}
                value={
                  metrics?.[key] !== undefined
                    ? metrics[key]!.toFixed(2)
                    : null
                }
              />
            ))}
          </div>
        )}
      </div>
    </AppLayout>
  );
}
