import { AlertTriangle } from "lucide-react";

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ title = "Something went wrong", message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-14 text-center px-6">
      <div className="h-10 w-10 rounded-full bg-bad-soft flex items-center justify-center">
        <AlertTriangle className="h-5 w-5 text-bad" />
      </div>
      <div>
        <p className="font-medium text-ink">{title}</p>
        <p className="text-sm text-ink-soft mt-1 max-w-sm">{message}</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 rounded-md border border-line-strong px-4 py-1.5 text-sm font-medium text-ink hover:bg-surface-wash transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
}
