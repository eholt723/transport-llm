import { useEffect, useRef, useState } from "react";
import {
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";
import docs from "./data/docs.json";
import { buildIndex, makeContext } from "./rag";

type ChatMsg = { role: "user" | "assistant" | "system"; content: string };

// Hardcode your production model (simple and stable)
const DEFAULT_MODEL_ID = "Llama-3-8B-Instruct-q4f16_1-MLC";

// One-time RAG index
const RAG_INDEX = buildIndex(docs as any);

// Base system prompt (concise, transport-focused)
const BASE_SYSTEM = `You are Transport LLM — a concise assistant about transportation history, terminology, engineering, and standards.
If context is provided, ground your answer in it. If the context is insufficient, say so briefly.`;

export default function App() {
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [status, setStatus] = useState("Loading model…");
  const [busy, setBusy] = useState(true);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([
    { role: "system", content: BASE_SYSTEM },
    { role: "assistant", content: "Welcome. Ask about rail history, signaling, or standards." },
  ]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [busy]);

  // Init engine once
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setBusy(true);
      try {
        const e = await CreateMLCEngine(DEFAULT_MODEL_ID, {
          initProgressCallback: (p: InitProgressReport) => setStatus(p.text),
        });
        if (!cancelled) {
          setEngine(e);
          setStatus("Ready");
        }
      } catch (err: any) {
        const msg = `Init error: ${err?.message ?? String(err)}`;
        setStatus(msg);
        setMessages((m) => [...m, { role: "assistant", content: msg }]);
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  async function send() {
    const userRaw = input.trim();
    if (!userRaw || !engine) return;

    // Build hidden RAG context as part of the USER message (not another system msg)
    let userWithContext = userRaw;
    const ctx = makeContext(userRaw, RAG_INDEX, 3, 1200);
    if (ctx) {
      userWithContext =
        `Use the transportation context below to answer the question. ` +
        `If the context is insufficient, say so briefly.\n\n` +
        `### Context\n${ctx}\n\n` +
        `### Question\n${userRaw}`;
    }

    // UI: show only the user's plain question
    setInput("");
    const uiNext: ChatMsg[] = [
      ...messages,
      { role: "user", content: userRaw },
      { role: "assistant", content: "" }, // placeholder for streaming
    ];
    setMessages(uiNext);
    setBusy(true);

    try {
      // Build API messages: exactly ONE system (base), then convo (no system),
      // and the augmented USER as the last message. Drop the trailing assistant placeholder.
      const convo = uiNext.filter((m) => m.role !== "system").slice(0, -1);
      const apiMessages: ChatMsg[] = [
        { role: "system", content: BASE_SYSTEM }, // first and only system
        ...convo.slice(0, -1),                    // prior chat (user/assistant)
        { role: "user", content: userWithContext } // augmented user
      ];

      const stream = await engine.chat.completions.create({
        stream: true,
        messages: apiMessages,
      });

      for await (const chunk of stream) {
        const delta = chunk?.choices?.[0]?.delta?.content ?? "";
        if (delta) {
          setMessages((curr) => {
            const updated = [...curr];
            const last = updated.length - 1;
            if (updated[last]?.role === "assistant") {
              updated[last] = {
                ...updated[last],
                content: (updated[last].content || "") + delta,
              };
            }
            return updated;
          });
        }
      }
    } catch (err: any) {
      setMessages((curr) => [
        ...curr,
        { role: "assistant", content: `Error: ${err?.message ?? String(err)}` },
      ]);
    } finally {
      setBusy(false);
    }
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
          <div className="status">{status}</div>
        </header>

        <main className="card chat">
          <div className="chat-scroll" ref={scrollRef}>
            {messages
              .filter((m) => m.role !== "system") // hide all system content from the UI
              .map((m, i) => (
                <div key={i} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                  {m.content}
                </div>
              ))}
          </div>

          <div className="input-row">
            <textarea
              ref={inputRef}
              placeholder={engine ? "Ask about rail history, signaling, etc…" : "Loading model…"}
              disabled={!engine || busy}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <button className="button" disabled={!engine || busy || !input.trim()} onClick={send}>
              {busy ? "…" : "Send"}
            </button>
          </div>
        </main>

        <footer className="status">
          Fully local via WebGPU · No server · RAG v1 enabled
        </footer>
      </div>
    </div>
  );
}
