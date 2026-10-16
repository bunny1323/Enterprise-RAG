import { api, toApiError } from "./api";
import type { EvaluationMetrics, HealthResponse } from "../types/system";

export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await api.get<HealthResponse>("/health");
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function getEvaluation(): Promise<EvaluationMetrics> {
  try {
    const res = await api.get<EvaluationMetrics>("/api/v1/evaluation");
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}
