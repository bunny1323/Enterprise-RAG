import { useCallback, useRef, useState } from "react";
import { streamChatCompletion } from "../services/chat";
import type { ChatResponse, Message } from "../types/chat";

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

interface AskOptions {
  industry?: string | null;
  documentId?: string | null;
  topK?: number;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const updateMessage = useCallback(
    (id: string, patch: Partial<Message>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...patch } : m))
      );
    },
    []
  );

  const askQuestion = useCallback(
    async (query: string, options: AskOptions = {}) => {
      setError(null);
      const userMessage: Message = {
        id: makeId(),
        role: "user",
        content: query,
      };
      const assistantId = makeId();
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
        reasoningStatus: [],
      };
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let accumulated = "";

      await streamChatCompletion(
        {
          query,
          industry: options.industry ?? null,
          document_id: options.documentId ?? null,
          top_k: options.topK ?? 5,
          stream: true,
        },
        {
          onReasoning: (data) => {
            const step =
              typeof data.step === "string" ? data.step : JSON.stringify(data);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      reasoningStatus: [...(m.reasoningStatus ?? []), step],
                    }
                  : m
              )
            );
          },
          onToken: (token) => {
            accumulated += token;
            updateMessage(assistantId, { content: accumulated });
          },
          onFinal: (data: ChatResponse) => {
            updateMessage(assistantId, {
              content: data.answer || accumulated,
              citations: data.citations,
              confidence: data.confidence_score,
              hallucinationStatus: data.hallucination_status,
              latencyMs: data.latency_ms,
              queryId: data.query_id,
              isStreaming: false,
            });
            setIsLoading(false);
          },
          onError: (err) => {
            updateMessage(assistantId, {
              isStreaming: false,
              error: err.message,
            });
            setError(err.message);
            setIsLoading(false);
          },
        },
        controller.signal
      );

      setIsLoading(false);
    },
    [updateMessage]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsLoading(false);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, askQuestion, isLoading, error, stop, clear };
}
