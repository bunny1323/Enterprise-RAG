import type { Citation, Message } from "../../types/chat";
import { API_BASE_URL } from "../../services/api";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { GroundingBadge } from "./GroundingBadge";
import { CitationCard } from "./CitationCard";
import { FeedbackButtons } from "./FeedbackButtons";
import { ReasoningTrace } from "./ReasoningTrace";
import { AlertTriangle } from "lucide-react";

interface ChatMessageProps {
  message: Message;
  onOpenCitation: (citation: Citation) => void;
}

export function ChatMessage({ message, onOpenCitation }: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-lg rounded-tr-sm bg-signal px-4 py-2.5 text-sm text-surface">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] rounded-lg rounded-tl-sm border border-line bg-surface px-4 py-3">
          <div className="flex items-start gap-2 text-bad text-sm">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Unable to generate an answer.</p>
              <p className="text-xs text-ink-soft mt-1">{message.error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] w-full rounded-lg rounded-tl-sm border border-line bg-surface px-4 py-3.5 space-y-3">
        {message.isStreaming && !message.content ? (
          <ReasoningTrace steps={message.reasoningStatus ?? []} active />
        ) : (
          <>
            <p className="text-sm text-ink whitespace-pre-wrap leading-relaxed">
              {message.content}
              {message.isStreaming && (
                <span className="inline-block w-1.5 h-4 bg-ink-faint ml-0.5 align-text-bottom animate-pulse" />
              )}
            </p>

            {!message.isStreaming && (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <GroundingBadge hallucinationStatus={message.hallucinationStatus} />
                  <ConfidenceBadge score={message.confidence} />
                  {message.latencyMs !== undefined && (
                    <span className="text-xs font-mono text-ink-faint">
                      {message.latencyMs}ms
                    </span>
                  )}
                </div>

                {message.citations && message.citations.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-ink-soft mb-2">
                      Sources
                    </p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {message.citations.map((c, i) => (
                        <CitationCard
                          key={i}
                          citation={c}
                          index={i}
                          onOpen={onOpenCitation}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {message.images && message.images.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-ink-soft mb-2">
                      Extracted Diagrams & Figures
                    </p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {message.images.map((img, i) => {
                        const token = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("erag_token") : null;
                        const fullUrl = `${API_BASE_URL}${img.url}${token ? `?token=${encodeURIComponent(token)}` : ""}`;
                        return (
                          <div key={i} className="rounded border border-line bg-surface p-2 space-y-1.5">
                            <img
                              src={fullUrl}
                              alt={img.caption || `Extracted diagram page ${img.page}`}
                              className="max-h-48 w-full object-contain rounded bg-black/5"
                              loading="lazy"
                            />
                            {img.caption && (
                              <p className="text-xs text-ink-soft line-clamp-2">{img.caption}</p>
                            )}
                            <p className="text-[10px] text-ink-faint">
                              {img.source ? `${img.source} • ` : ""}Page {img.page}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <FeedbackButtons queryId={message.queryId} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
