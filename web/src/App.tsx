import { useEffect, useRef, useState } from "react";
import {
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";

import { retrieve } from "./rag/retriever";
import { buildAugmentedPrompt } from "./rag/augment";
// Optional: quick retrieval sanity-check UI
// import DebugPanel from "./rag/debug/DebugPanel";

type ChatMsg = { role: "user" | "assistant" | "system"; content: string };

// Hardcode your production model (simple and stable)
const DEFAULT_MODEL_ID = "Llama-3-8B-Instruct-q4f16_1-MLC";

// Base system prompt (concise, transport-focused)
const BASE_SYSTEM = `You are Transport LLM — a concise assistant focused on rail operations, automotive engineering, intelligent transit, and transport standards.
If context is provided, ground your answer in it. If the context is insufficient, say so briefly.
When you use provided context, cite sources by [title]. If a user corrects you, accept the correction politely.`;

export default function App() {
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [status, setStatus] = useState("Loading model…");
  const [busy, setBusy] = useState(true);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([
    { role: "system", content: BASE_SYSTEM },
    { role: "assistant", content: "Welcome. Ask about rail operations, automotive systems, public transit, or transportation standards." },
  ]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  // Keep cursor focus when ready
  useEffect(() => {
    if (!busy) inputRef.current?.focus();
  }, [busy]);

  // Initialize the MLC engine once
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
    return () => {
      cancelled = true;
    };
  }, []);

  async function send() {
    const userRaw = input.trim();
    if (!userRaw || !engine) return;

    // 1) Retrieve top-k chunks using the browser-side index
    //    (requires web/public/rag/{index.json,embeddings.f32,manifest.json})
    let augmentedUser = userRaw;
    try {
      const k = 5;
      const ret = await retrieve(userRaw, {
        k,
        domains: [], // e.g., ["rail","auto"] or leave empty for all
        domainWeights: { rail: 1.0, auto: 0.95, transit: 0.95, standards: 1.0 },
        mmr: { lambda: 0.7, fetchK: Math.max(40, k * 4) },
      });

      // 2) Build an augmented prompt within a token budget
      const augmented = buildAugmentedPrompt(userRaw, ret.topK, {
        maxTokens: 1200,         // adjust to fit your model's context budget
        charsPerToken: 4,        // heuristic
        header: "Context (use for citations):",
      });

      // 3) Use the augmented prompt as the final user message for generation
      augmentedUser = augmented;
    } catch (e: any) {
      // If retrieval fails for any reason, proceed without augmentation
      console.warn("RAG retrieval failed:", e);
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
      // Compose the API messages:
      // - exactly ONE system (BASE_SYSTEM)
      // - prior user/assistant turns (no system)
      // - augmented user last
      const prior = uiNext.filter((m) => m.role !== "system").slice(0, -1);
      const apiMessages: ChatMsg[] = [
        { role: "system", content: BASE_SYSTEM },
        ...prior.slice(0, -1), // all previous user/assistant messages
        { role: "user", content: augmentedUser },
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
          <h1>Transportation LLM — Edge AI Demo</h1>
          <div className="status">{status}</div>
        </header>

        <main className="card chat">
          {/* Optional debug panel for RAG sanity checks */}
          {/* <div className="mb-3"><DebugPanel /></div> */}

          <div className="chat-scroll" ref={scrollRef}>
            {messages
              .filter((m) => m.role !== "system") // hide system content from the UI
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
