const STAGES = ["Dense", "BM25", "Graph", "Image"];

export function RetrievalPipeline() {
  return (
    <div className="rounded-lg border border-line bg-surface px-5 py-6">
      <p className="text-sm font-medium text-ink mb-4">Retrieval pipeline</p>
      <div className="flex flex-col sm:flex-row items-center gap-3 sm:gap-0 text-xs font-mono">
        <div className="flex flex-col gap-1.5">
          {STAGES.map((s) => (
            <div
              key={s}
              className="rounded-md border border-line-strong bg-surface-sunk px-3 py-1.5 text-center text-ink-soft"
            >
              {s} search
            </div>
          ))}
        </div>
        <PipelineArrow />
        <PipelineNode label="RRF fusion" />
        <PipelineArrow />
        <PipelineNode label="Reranking" />
        <PipelineArrow />
        <PipelineNode label="Results" emphasis />
      </div>
    </div>
  );
}

function PipelineArrow() {
  return (
    <div className="hidden sm:block h-px w-6 bg-line-strong mx-2" aria-hidden="true" />
  );
}

function PipelineNode({ label, emphasis }: { label: string; emphasis?: boolean }) {
  return (
    <div
      className={`rounded-md border px-3 py-2 text-center whitespace-nowrap ${
        emphasis
          ? "border-signal bg-signal-soft text-signal-dark font-medium"
          : "border-line-strong bg-surface-sunk text-ink-soft"
      }`}
    >
      {label}
    </div>
  );
}
