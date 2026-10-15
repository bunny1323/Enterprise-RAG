import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;

if (!API_BASE_URL) {
  // Fail loudly in dev rather than silently hitting the wrong origin.
  // eslint-disable-next-line no-console
  console.warn(
    "VITE_API_BASE_URL is not set. Create a .env file (see .env.example)."
  );
}

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Attach an auth token if one has been issued to this browser session.
// The frontend does not implement login; it only forwards a token if one
// exists, in case the backend enforces JWT/OIDC auth.
api.interceptors.request.use((config) => {
  const token = sessionStorage.getItem("erag_token");
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

export function toApiError(err: unknown): ApiError {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    if (!err.response) {
      return new ApiError(
        "Backend unavailable. Check whether the Enterprise-RAG backend is running.",
        undefined
      );
    }
    const detail =
      (err.response.data as { detail?: string; message?: string } | undefined)
        ?.detail ??
      (err.response.data as { detail?: string; message?: string } | undefined)
        ?.message;
    return new ApiError(detail ?? err.message, status);
  }
  return new ApiError("Something went wrong. Please try again.");
}

export { API_BASE_URL };
