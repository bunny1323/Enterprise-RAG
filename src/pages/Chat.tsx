import { useState } from "react";
import { Plus } from "lucide-react";
import { AppLayout } from "../components/layout/AppLayout";
import { ChatWindow } from "../components/chat/ChatWindow";
import { SourceDrawer } from "../components/chat/SourceDrawer";
import { useChat } from "../hooks/useChat";
import { useDocuments } from "../hooks/useDocuments";
import type { Citation } from "../types/chat";
import { INDUSTRY_OPTIONS } from "../types/document";

interface ConversationSummary {
  id: string;
  title: string;
}

export default function Chat() {
  const { messages, askQuestion, isLoading, stop, clear } = useChat();
  const { documents } = useDocuments();
  const [industry, setIndustry] = useState<string>("");
  const [documentId, setDocumentId] = useState<string>("");
  const [topK, setTopK] = useState(5);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  // Conversation history is not yet persisted by the backend, so this list
  // is held only in frontend state for the duration of the session.
  const [conversations, setConversations] = useState<ConversationSummary[]>(
    []
  );
  const [activeConversation, setActiveConversation] = useState<string | null>(
    null
  );

  const handleSend = (query: string) => {
    if (!activeConversation) {
      const id = `${Date.now()}`;
      setConversations((prev) => [
        { id, title: query.slice(0, 40) },
        ...prev,
      ]);
      setActiveConversation(id);
    }
    askQuestion(query, {
      industry: industry || null,
      documentId: documentId || null,
      topK,
    });
  };

  const startNewChat = () => {
    clear();
    setActiveConversation(null);
  };

  return (
    <AppLayout title="Ask AI" fullBleed>
      <div className="flex h-full min-h-0">
        {/* Conversation history */}
        <div className="hidden lg:flex w-56 shrink-0 flex-col border-r border-line bg-surface">
          <div className="p-3">
            <button
              onClick={startNewChat}
              className="flex w-full items-center gap-1.5 rounded-md border border-line-strong px-3 py-2 text-sm text-ink hover:bg-surface-wash transition-colors"
            >
              <Plus className="h-3.5 w-3.5" />
              New conversation
            </button>
          </div>
          <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-0.5">
            {conversations.length === 0 ? (
              <p className="text-xs text-ink-faint px-1 pt-2">
                No conversations yet.
              </p>
            ) : (
              conversations.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setActiveConversation(c.id)}
                  className={`w-full truncate rounded-md px-2.5 py-1.5 text-left text-sm transition-colors ${
                    activeConversation === c.id
                      ? "bg-signal-soft text-signal-dark"
                      : "text-ink-soft hover:bg-surface-wash"
                  }`}
                >
                  {c.title}
                </button>
              ))
            )}
          </div>
        </div>

        {/* Chat area */}
        <div className="flex flex-1 flex-col min-w-0">
          <div className="flex flex-wrap items-center gap-3 border-b border-line bg-surface px-4 py-2.5">
            <label className="flex items-center gap-1.5 text-xs text-ink-soft">
              Industry
              <select
                value={industry}
                onChange={(e) => setIndustry(e.target.value)}
                className="rounded-md border border-line-strong bg-surface-sunk px-2 py-1 text-xs text-ink"
              >
                <option value="">All industries</option>
                {INDUSTRY_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt.charAt(0).toUpperCase() + opt.slice(1)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-xs text-ink-soft">
              Document
              <select
                value={documentId}
                onChange={(e) => setDocumentId(e.target.value)}
                className="rounded-md border border-line-strong bg-surface-sunk px-2 py-1 text-xs text-ink max-w-[10rem] truncate"
              >
                <option value="">All documents</option>
                {documents.map((d) => (
                  <option key={d.document_id} value={d.document_id}>
                    {d.title || d.file_name || d.document_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-xs text-ink-soft">
              Sources
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="rounded-md border border-line-strong bg-surface-sunk px-2 py-1 text-xs text-ink"
              >
                {[3, 5, 8, 10].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            onSend={handleSend}
            onStop={stop}
            onOpenCitation={setActiveCitation}
          />
        </div>
      </div>

      <SourceDrawer
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />
    </AppLayout>
  );
}
