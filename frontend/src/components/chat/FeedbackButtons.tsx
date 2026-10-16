import { useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { submitFeedback } from "../../services/chat";

export function FeedbackButtons({ queryId }: { queryId?: string }) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!queryId) return null;

  const send = async (value: 1 | -1) => {
    setRating(value);
    setError(null);
    try {
      await submitFeedback(queryId, value);
    } catch {
      setError("Couldn't submit feedback.");
    }
  };

  return (
    <div className="flex items-center gap-3 pt-1">
      <span className="text-xs text-ink-soft">Was this answer helpful?</span>
      <button
        onClick={() => send(1)}
        aria-label="Thumbs up"
        className={`transition-colors ${
          rating === 1 ? "text-good" : "text-ink-faint hover:text-ink"
        }`}
      >
        <ThumbsUp className="h-4 w-4" />
      </button>
      <button
        onClick={() => send(-1)}
        aria-label="Thumbs down"
        className={`transition-colors ${
          rating === -1 ? "text-bad" : "text-ink-faint hover:text-ink"
        }`}
      >
        <ThumbsDown className="h-4 w-4" />
      </button>
      {error && <span className="text-xs text-bad">{error}</span>}
    </div>
  );
}
