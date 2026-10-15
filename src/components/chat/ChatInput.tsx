import { useState } from "react";
import type { KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";

interface ChatInputProps {
  onSend: (query: string) => void;
  disabled?: boolean;
  onStop?: () => void;
}

export function ChatInput({ onSend, disabled, onStop }: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex items-end gap-2 border-t border-line bg-surface px-4 py-3">
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question…"
        rows={1}
        className="flex-1 resize-none rounded-md border border-line-strong bg-surface-sunk px-3 py-2 text-sm text-ink focus:border-signal max-h-32"
      />
      <button
        onClick={disabled ? onStop : submit}
        disabled={disabled ? !onStop : !value.trim()}
        aria-label={disabled ? "Stop generating" : "Send"}
        className="flex items-center justify-center h-9 w-9 rounded-md bg-signal text-surface hover:bg-signal-dark transition-colors disabled:opacity-50 shrink-0"
      >
        {disabled ? <Square className="h-3.5 w-3.5 fill-current" /> : <Send className="h-4 w-4" />}
      </button>
    </div>
  );
}
