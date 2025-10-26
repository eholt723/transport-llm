import { useEffect, useRef, useState } from "react";

type ChatMsg = { role: "user" | "assistant"; content: string };

export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([
    { role: "assistant", content: "Welcome to Transport LLM — Edge AI Demo." },
    { role: "assistant", content: "Stage 4 complete: structure cleaned and ready for chat." },
  ]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  // Keep cursor focused
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function send() {
    const content = input.trim();
    if (!content) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content }]);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app-frame">
      <div className="app-wrap">
        <header className="header">
          <h1>Transport LLM — Edge AI Demo</h1>
          <div className="status">Stage 4: Clean layout ready</div>
        </header>

        <main className="card chat">
          <div className="chat-scroll" ref={scrollRef}>
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                {m.content}
              </div>
            ))}
          </div>

          <div className="input-row">
            <textarea
              ref={inputRef}
              placeholder="Type a message…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <button className="button" onClick={send} disabled={!input.trim()}>
              Send
            </button>
          </div>
        </main>

        <footer className="status">UI ready for model integration.</footer>
      </div>
    </div>
  );
}
