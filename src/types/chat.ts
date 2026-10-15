export interface Citation {
  document_id: string;
  document_name?: string;
  page_number?: number;
  bbox?: [number, number, number, number];
  snippet: string;
  weight?: number;
  chunk_id?: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  confidence_score?: number;
  hallucination_status?: boolean;
  reasoning_trace?: string[];
  latency_ms?: number;
  query_id?: string;
}

export interface ChatRequest {
  query: string;
  industry?: string | null;
  document_id?: string | null;
  top_k?: number;
  stream?: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  confidence?: number;
  hallucinationStatus?: boolean;
  latencyMs?: number;
  queryId?: string;
  reasoningStatus?: string[];
  isStreaming?: boolean;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  messages: Message[];
}

/** SSE event payload shapes documented by the backend. */
export interface ReasoningEvent {
  step: string;
  status?: string;
  [key: string]: unknown;
}

export interface TokenEvent {
  token: string;
}
