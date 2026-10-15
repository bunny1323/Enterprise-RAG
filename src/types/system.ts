export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy" | string;
  services: Record<string, string>;
  version?: string;
}

export interface EvaluationMetrics {
  faithfulness?: number;
  context_precision?: number;
  context_recall?: number;
  layout_extraction_accuracy?: number;
  [key: string]: number | undefined;
}
