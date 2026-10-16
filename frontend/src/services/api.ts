import axios from "axios";

// Normalize trailing slash and provide safe local dev default if omitted
const rawBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
const API_BASE_URL = rawBaseUrl ? rawBaseUrl.replace(/\/+$/, "") : "";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Attach an auth token if one has been issued to this browser session.
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
      if (err.code === "ECONNABORTED" || err.message?.includes("timeout")) {
        return new ApiError("Request timed out. The backend took too long to respond.", 408);
      }
      return new ApiError(
        "Backend unavailable. Please verify that the Enterprise-RAG backend is running.",
        undefined
      );
    }

    const data = err.response.data as { detail?: string | { msg?: string }[]; message?: string } | undefined;
    let detailMessage = "";
    if (typeof data?.detail === "string") {
      detailMessage = data.detail;
    } else if (Array.isArray(data?.detail) && data?.detail[0]?.msg) {
      detailMessage = data.detail.map((d) => d.msg).join("; ");
    } else if (data?.message) {
      detailMessage = data.message;
    }

    if (detailMessage) {
      return new ApiError(detailMessage, status);
    }

    switch (status) {
      case 400:
        return new ApiError("Invalid request. Please check your inputs.", status);
      case 401:
        return new ApiError("Unauthorized. Please check your authentication token.", status);
      case 403:
        return new ApiError("Forbidden. You do not have permission to perform this action.", status);
      case 404:
        return new ApiError("Requested resource was not found.", status);
      case 413:
        return new ApiError("The uploaded file exceeds the 100 MB maximum size limit.", status);
      case 429:
        return new ApiError("Rate limit exceeded. Please wait a moment before trying again.", status);
      case 500:
      case 502:
      case 503:
      case 504:
        return new ApiError("Backend service error. Please try again later.", status);
      default:
        return new ApiError(err.message || "An unexpected error occurred.", status);
    }
  }

  if (err instanceof Error) {
    return new ApiError(err.message);
  }

  return new ApiError("Something went wrong. Please try again.");
}

export { API_BASE_URL };
