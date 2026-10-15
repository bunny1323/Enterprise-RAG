export interface SearchRequest {
  query: string;
  industry?: string | null;
  top_k?: number;
}

/**
 * The architecture document notes the exact /search response schema is not
 * fully specified — verify against the running backend's OpenAPI schema
 * (/docs) and adjust this shape before relying on fields beyond these.
 */
export interface RetrievalChunk {
  document_id: string;
  document_name?: string;
  page_number?: number;
  chunk_id?: string;
  content?: string;
  score?: number;
  retrieval_type?: "dense" | "bm25" | "graph" | "image" | string;
}

export type SearchResponse = RetrievalChunk[];
