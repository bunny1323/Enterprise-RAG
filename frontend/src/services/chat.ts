import { API_BASE_URL, toApiError } from "./api";
import type { ChatRequest, ChatResponse } from "../types/chat";

/**
 * Enterprise-RAG conversational RAG endpoint:
 * POST /api/v1/chat
 */
const CHAT_PATH = "/api/v1/chat";

export interface StreamHandlers {
  onReasoning?: (data: Record<string, unknown>) => void;
  onToken?: (token: string) => void;
  onFinal?: (data: ChatResponse) => void;
  onError?: (error: Error) => void;
}

export function normalizeChatResponse(data: any): ChatResponse {
  return {
    answer: data.answer || "No response generated.",
    citations: (data.citations || []).map((c: any) => ({
      document_id: c.document_id || "",
      document_name: c.file_name || c.document_name || c.document_id || "Document",
      page_number: c.page_number,
      snippet: c.snippet || "",
      chunk_id: c.chunk_id,
      weight: c.weight,
      bbox: c.bbox,
    })),
    confidence_score:
      typeof data.confidence_score === "number"
        ? data.confidence_score
        : data.confidence === "HIGH"
        ? 0.95
        : data.confidence === "MEDIUM"
        ? 0.75
        : 0.5,
    confidence: data.confidence,
    hallucination_status: data.verification_status === "UNSUPPORTED",
    verification_status: data.verification_status,
    reasoning_trace: [
      `Intent: ${data.intent || "GENERAL_QA"}`,
      `Strategy: ${data.retrieval_strategy || "hybrid"}`,
      `Evidence chunks: ${data.evidence?.length ?? 0}`,
      `Verification: ${data.verification_status || "SUPPORTED"}`,
    ],
    latency_ms: data.latency_ms,
    query_id: data.trace_id,
    images: data.images,
  };
}

export async function streamChatCompletion(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  const url = `${API_BASE_URL}${CHAT_PATH}`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  // Reuse the authentication token from sessionStorage (consistent with Axios interceptor in api.ts)
  const token = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("erag_token") : null;
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        query: request.query,
        top_k: request.top_k ?? 10,
        retrieval_profile: "default",
      }),
      signal,
    });
  } catch (err: unknown) {
    if ((err as Error)?.name === "AbortError") return;
    handlers.onError?.(
      new Error(
        "Backend unavailable. Please verify that the Enterprise-RAG backend is running."
      )
    );
    return;
  }

  if (response.status === 401) {
    handlers.onError?.(
      new Error("Authentication required or session expired. Please sign in again.")
    );
    return;
  }

  if (response.status === 403) {
    handlers.onError?.(
      new Error("Access forbidden: you do not have permission to query this resource.")
    );
    return;
  }

  if (!response.ok || !response.body) {
    let errorDetail = `Unable to generate an answer (status ${response.status}).`;
    try {
      const errJson = await response.json();
      if (errJson?.detail) errorDetail = String(errJson.detail);
    } catch {
      // ignore
    }
    handlers.onError?.(new Error(errorDetail));
    return;
  }

  const contentType = response.headers.get("content-type") || "";

  // ── Handle standard JSON response (FastAPI default) ──────────────────────
  if (!contentType.includes("text/event-stream")) {
    try {
      const raw = await response.json();
      const finalResp = normalizeChatResponse(raw);

      // Smooth progressive token stream effect for user feedback
      const words = (finalResp.answer || "").split(" ");
      let accumulated = "";
      for (let i = 0; i < words.length; i++) {
        if (signal?.aborted) return;
        const part = (i > 0 ? " " : "") + words[i];
        accumulated += part;
        handlers.onToken?.(part);
        if (i % 6 === 0 && i < words.length - 1) {
          await new Promise((r) => setTimeout(r, 12));
        }
      }

      handlers.onFinal?.(finalResp);
      return;
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        handlers.onError?.(toApiError(err));
      }
      return;
    }
  }

  // ── Handle SSE event-stream if streaming is active ───────────────────────
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const rawEvent of events) {
        const lines = rawEvent.split("\n");
        let eventName = "message";
        let dataLine = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLine += line.slice(5).trim();
          }
        }
        if (!dataLine) continue;

        let parsed: unknown;
        try {
          parsed = JSON.parse(dataLine);
        } catch {
          continue;
        }

        if (eventName === "reasoning") {
          handlers.onReasoning?.(parsed as Record<string, unknown>);
        } else if (eventName === "token") {
          const token = (parsed as { token?: string }).token;
          if (token) handlers.onToken?.(token);
        } else if (eventName === "final") {
          handlers.onFinal?.(normalizeChatResponse(parsed));
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      handlers.onError?.(toApiError(err));
    }
  }
}

/** Non-streaming fallback */
export async function askQuestionOnce(
  request: ChatRequest
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL}${CHAT_PATH}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: request.query,
      top_k: request.top_k ?? 10,
    }),
  });
  if (!res.ok) {
    throw new Error(`Unable to generate an answer (status ${res.status}).`);
  }
  const data = await res.json();
  return normalizeChatResponse(data);
}

export async function submitFeedback(
  queryId: string,
  rating: 1 | 0 | -1,
  comment?: string
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_id: queryId, rating, comment }),
  });
  if (!res.ok) {
    throw new Error("Unable to submit feedback.");
  }
}
