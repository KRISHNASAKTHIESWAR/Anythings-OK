"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { streamChat } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

// ── Icons ────────────────────────────────────────────────────────────────────

function IconSend() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function IconGraph() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="5" r="3"/>
      <circle cx="19" cy="19" r="3"/>
      <circle cx="5" cy="19" r="3"/>
      <line x1="12" y1="8" x2="19" y2="16"/>
      <line x1="12" y1="8" x2="5" y2="16"/>
    </svg>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ChatArea() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const sendMessage = useCallback(async () => {
    const query = input.trim();
    if (!query || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: query };
    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: Message = { id: assistantId, role: "assistant", content: "", streaming: true };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    await streamChat(
      query,
      (token) => setMessages((prev) =>
        prev.map((m) => m.id === assistantId ? { ...m, content: m.content + token } : m)
      ),
      () => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantId ? { ...m, streaming: false } : m)
        );
        setIsStreaming(false);
      },
      (err) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: `Error: ${err}`, streaming: false } : m
          )
        );
        setIsStreaming(false);
      }
    );
  }, [input, isStreaming]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
        background: "var(--bg-primary)",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "14px 24px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          background: "var(--bg-primary)",
        }}
      >
        <div
          style={{
            width: "7px", height: "7px", borderRadius: "50%",
            background: "#16a34a",
            boxShadow: "0 0 6px rgba(22,163,74,0.6)",
          }}
        />
        <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>
          Knowledge Graph Chat
        </span>
        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
          GraphRAG + Ollama
        </span>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "24px 0",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {messages.length === 0 ? (
          <WelcomeScreen />
        ) : (
          messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div
        style={{
          padding: "14px 20px 18px",
          borderTop: "1px solid var(--border)",
          background: "var(--bg-primary)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: "10px",
            background: "var(--input-bg)",
            border: `1px solid ${isStreaming ? "var(--border-bright)" : "var(--border)"}`,
            borderRadius: "12px",
            padding: "11px 14px",
            transition: "border-color 0.2s ease",
            boxShadow: "var(--shadow)",
          }}
        >
          <textarea
            ref={textareaRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your documents..."
            disabled={isStreaming}
            rows={1}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              resize: "none",
              color: "var(--text-primary)",
              fontSize: "14px",
              lineHeight: "1.5",
              maxHeight: "160px",
              fontFamily: "inherit",
            }}
          />
          <button
            id="send-button"
            onClick={sendMessage}
            disabled={isStreaming || !input.trim()}
            style={{
              width: "34px",
              height: "34px",
              borderRadius: "8px",
              border: "none",
              cursor: isStreaming || !input.trim() ? "not-allowed" : "pointer",
              background: isStreaming || !input.trim() ? "var(--bg-card-hover)" : "var(--accent)",
              color: isStreaming || !input.trim() ? "var(--text-muted)" : "#ffffff",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "all 0.15s ease",
            }}
          >
            {isStreaming ? <div className="spinner" /> : <IconSend />}
          </button>
        </div>
        <div style={{ textAlign: "center", marginTop: "7px" }}>
          <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
            Shift+Enter for new line · Enter to send
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function WelcomeScreen() {
  const suggestions = [
    "What are the main topics in my documents?",
    "Summarize the key entities and their relationships.",
    "What communities were detected in the knowledge graph?",
    "Find connections between the uploaded documents.",
  ];

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "40px 24px",
        gap: "28px",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            width: "44px",
            height: "44px",
            borderRadius: "12px",
            background: "var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 18px",
            color: "white",
            boxShadow: "0 4px 12px rgba(210,143,75,0.3)",
          }}
        >
          <IconGraph />
        </div>
        <h1
          style={{
            fontSize: "22px",
            fontWeight: 700,
            color: "var(--text-primary)",
            letterSpacing: "-0.4px",
            marginBottom: "8px",
          }}
        >
          Ready to explore your knowledge graph
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "13px", maxWidth: "400px", lineHeight: "1.65" }}>
          Upload documents using the sidebar, then ask questions. The system uses GraphRAG
          — combining entity graphs and community summaries for deep answers.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "10px",
          maxWidth: "540px",
          width: "100%",
        }}
      >
        {suggestions.map((s, i) => (
          <SuggestionCard key={i} text={s} />
        ))}
      </div>
    </div>
  );
}

function SuggestionCard({ text }: { text: string }) {
  return (
    <div
      style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border)",
        borderRadius: "10px",
        padding: "13px 15px",
        fontSize: "13px",
        color: "var(--text-secondary)",
        cursor: "pointer",
        transition: "all 0.15s ease",
        lineHeight: "1.45",
        boxShadow: "var(--shadow)",
      }}
      onMouseEnter={(e) => {
        const el = e.currentTarget as HTMLDivElement;
        el.style.borderColor = "var(--border-bright)";
        el.style.color = "var(--text-primary)";
        el.style.boxShadow = "var(--shadow-lg)";
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget as HTMLDivElement;
        el.style.borderColor = "var(--border)";
        el.style.color = "var(--text-secondary)";
        el.style.boxShadow = "var(--shadow)";
      }}
      onClick={() => {
        const ta = document.getElementById("chat-input") as HTMLTextAreaElement | null;
        if (ta) {
          ta.value = text;
          ta.dispatchEvent(new Event("input", { bubbles: true }));
          ta.focus();
        }
      }}
    >
      {text}
    </div>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div
      className="slide-up"
      style={{
        padding: "6px 24px",
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        gap: "10px",
      }}
    >
      {/* Avatar */}
      {!isUser && (
        <div
          style={{
            width: "28px",
            height: "28px",
            borderRadius: "8px",
            background: "var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            marginTop: "2px",
          }}
        >
          <IconGraph />
        </div>
      )}

      <div
        style={{
          maxWidth: "70%",
          background: isUser ? "var(--bg-card)" : "transparent",
          border: isUser ? "1px solid var(--border)" : "none",
          borderRadius: isUser ? "12px 12px 4px 12px" : "4px 12px 12px 12px",
          padding: isUser ? "10px 14px" : "4px 0",
          fontSize: "14px",
          lineHeight: "1.65",
          color: "var(--text-primary)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          boxShadow: isUser ? "var(--shadow)" : "none",
        }}
        className={message.streaming && message.content ? "cursor-blink" : ""}
      >
        {message.content || (
          message.streaming ? (
            <span style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "13px" }}>
              <div className="spinner" /> Thinking...
            </span>
          ) : null
        )}
      </div>

      {isUser && (
        <div
          style={{
            width: "28px",
            height: "28px",
            borderRadius: "8px",
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            marginTop: "2px",
            fontSize: "11px",
            fontWeight: 700,
            color: "var(--text-muted)",
            boxShadow: "var(--shadow)",
          }}
        >
          U
        </div>
      )}
    </div>
  );
}
