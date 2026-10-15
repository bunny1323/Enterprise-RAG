import { api, toApiError } from "./api";
import type { SearchRequest, SearchResponse } from "../types/search";

const SEARCH_PATH = "/api/v1/search";

export async function runSearch(
  request: SearchRequest
): Promise<SearchResponse> {
  try {
    const res = await api.post<SearchResponse>(SEARCH_PATH, request);
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}
