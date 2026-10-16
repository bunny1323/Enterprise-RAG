import { api, toApiError } from "./api";
import type { SearchRequest, RetrievalChunk } from "../types/search";

const SEARCH_PATH = "/api/v1/search";

export async function runSearch(
  request: SearchRequest
): Promise<RetrievalChunk[]> {
  try {
    const res = await api.post<any>(SEARCH_PATH, {
      query: request.query,
      top_k: request.top_k ?? 10,
    });
    const data = res.data;
    const rawList = Array.isArray(data)
      ? data
      : Array.isArray(data?.results)
      ? data.results
      : [];

    return rawList.map((item: any) => ({
      document_id: item.document_id || "",
      document_name: item.file_name || item.document_name || item.document_id || "Document",
      page_number: item.page_number,
      chunk_id: item.chunk_id,
      content: item.content,
      score: item.score,
      retrieval_type: item.chunk_type || item.retrieval_type || "dense",
    }));
  } catch (err) {
    throw toApiError(err);
  }
}
