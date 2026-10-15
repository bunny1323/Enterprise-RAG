import { API_BASE_URL, toApiError } from "./api";
import type { ChatRequest, ChatResponse } from "../types/chat";

/**
 * Endpoint note: the handoff docs disagree on singular vs plural
 * ("/chat/completion" vs "/chat/completions"). The detailed spec uses the
 * plural form — verify against the running backend's /docs before relying
 * on this in production.
 */
const CHAT_PATH = "/api/v1/chat/completions";

export interface StreamHandlers {
  onReasoning?: (data: Record<string, unknown>) => void;
  onToken?: (token: string) => void;
  onFinal?: (data: ChatResponse) => void;
  onError?: (error: Error) => void;
}

/**
 * Parses a standard SSE stream of the form:
 *   event: <name>
 *   data: <json>
 *   \n
 * The backend documents three event types: reasoning, token, final.
 */
export async function streamChatCompletion(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL ?? ""}${CHAT_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...request, stream: true }),
      signal,
    });
  } catch {
    handlers.onError?.(
      new Error(
        "Backend unavailable. Check whether the Enterprise-RAG backend is running."
      )
    );
    return;
  }

  if (!response.ok || !response.body) {
    handlers.onError?.(
      new Error(`Unable to generate an answer (status ${response.status}).`)
    );
    return;
  }

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
          handlers.onFinal?.(parsed as ChatResponse);
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      handlers.onError?.(toApiError(err));
    }
  }
}

/** Non-streaming fallback, for backends that don't stream. */
export async function askQuestionOnce(
  request: ChatRequest
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL ?? ""}${CHAT_PATH}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...request, stream: false }),
  });
  if (!res.ok) {
    throw new Error(`Unable to generate an answer (status ${res.status}).`);
  }
  return res.json();
}

export async function submitFeedback(
  queryId: string,
  rating: 1 | 0 | -1,
  comment?: string
): Promise<void> {
  const res = await fetch(`${API_BASE_URL ?? ""}/api/v1/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_id: queryId, rating, comment }),
  });
  if (!res.ok) {
    throw new Error("Unable to submit feedback.");
  }
}
