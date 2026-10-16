import { useEffect, useRef } from "react";
import type { Citation, Message } from "../../types/chat";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "../common/EmptyState";
import { Sparkles } from "lucide-react";

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  onSend: (query: string) => void;
  onStop?: () => void;
  onOpenCitation: (citation: Citation) => void;
}

export function ChatWindow({ messages, isLoading, onSend, onStop, onOpenCitation }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 overflow-y-auto px-4 py-5 space-y-4">
        {messages.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title="Ask your enterprise knowledge base"
            description='Try: "What is the maintenance procedure?"'
          />
        ) : (
          messages.map((m) => (
            <ChatMessage key={m.id} message={m} onOpenCitation={onOpenCitation} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
      <ChatInput onSend={onSend} onStop={onStop} disabled={isLoading} />
    </div>
  );
}
