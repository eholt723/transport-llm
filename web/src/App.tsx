import { useEffect, useRef, useState } from "react";
import {
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";

import { retrieve } from "./rag/retriever";
import { buildAugmentedPrompt } from "./rag/augment";

type ChatMsg = { role: "user" | "assistant" | "system"; content: string };

// Hardcode production model
const DEFAULT_MODEL_ID = "Llama-3-8B-Instruct-q4f16_1-MLC";

// Base system prompt
const BASE_SYSTEM = `You are Transport LLM — a concise assistant focused on rail operations, automotive engineering, intelligent transit, and transport standards.
If context is provided, ground your answer in it. If the context is insufficient, say so briefly.
When you use provided context, cite sources by [title]. If a user corrects you, accept the correction politely.`;

export default function App() {
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [status, setStatus] = useState("Loading model…");
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMsg[]>([
    { role: "system", content: BASE_SYSTEM },
    {
      role: "assistant",
      content:
        "Welcome. I’m designed to assist with rail operations, vehicle systems, public transit engineering, and transportation standards.",
    },
  ]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Flags
  const hasInitError = status.startsWith("Init error");
  const hasRuntimeError = !!runtimeError;

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  // Focus input
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Re-focus
  useEffect(() => {
    if (!busy) inputRef.current?.focus();
  }, [busy]);

  // Initialize MLC
  useEffect(() => {
    let cancelled = false;

    (async () => {
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
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  async function send() {
    const userRaw = input.trim();
    if (!userRaw || !engine) return;

    let augmentedUser = userRaw;

    try {
      const k = 5;
      const ret = await retrieve(userRaw, {
        k,
        domains: [],
        domainWeights: { rail: 1.0, auto: 0.95, transit: 0.95, standards: 1.0 },
        mmr: { lambda: 0.7, fetchK: Math.max(40, k * 4) },
      });

      const augmented = buildAugmentedPrompt(userRaw, ret.topK, {
        maxTokens: 1200,
        charsPerToken: 4,
        header: "Context (use for citations):",
      });

      augmentedUser = augmented;
    } catch (e: any) {
      console.warn("RAG retrieval failed:", e);
    }

    setInput("");
    const uiNext: ChatMsg[] = [
      ...messages,
      { role: "user", content: userRaw },
      { role: "assistant", content: "" },
    ];
    setMessages(uiNext);
    setBusy(true);

    try {
      const prior = uiNext.filter((m) => m.role !== "system").slice(0, -1);
      const apiMessages: ChatMsg[] = [
        { role: "system", content: BASE_SYSTEM },
        ...prior.slice(0, -1),
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
      console.error("Runtime error from WebLLM:", err);

      const friendly =
        "The model runtime encountered an error and stopped. This usually means your device or browser does not fully support WebGPU for this model, or ran out of GPU memory. Try Chrome or Edge on a desktop/laptop with a modern GPU. Mobile browsers may not support WebGPU.";

      setRuntimeError(friendly);

      setMessages((curr) => [
        ...curr,
        { role: "assistant", content: friendly },
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

  const canSend = !!engine && !busy && !!input.trim();

  return (
    <div className="app-frame">
      <div className="app-wrap">
        <header className="header">
          <h1>Transportation LLM — Edge AI Demo</h1>
          <div className="status">{status}</div>
        </header>

        {/* Unified WebGPU warning banner */}
        {(hasInitError || hasRuntimeError) && (
          <div className="warning-banner">
            Your device may not support WebGPU.
            <br />
            This demo requires a WebGPU-compatible GPU and browser.
            <br />
            If you see initialization errors, try Chrome or Edge on a desktop/laptop with a modern GPU.
            <br />
            Mobile browsers may not support WebGPU.
          </div>
        )}

        <main className="card chat">
          <div className="chat-scroll" ref={scrollRef}>
            {messages
              .filter((m) => m.role !== "system")
              .map((m, i) => (
                <div key={i} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                  {m.content}
                </div>
              ))}
          </div>

          <div className="input-row">
            <textarea
              ref={inputRef}
              placeholder={
                engine ? "Ask about rail history, signaling, etc…" : "Model loading… you can still type"
              }
              disabled={busy}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <button className="button" disabled={!canSend} onClick={send}>
              {busy ? "…" : "Send"}
            </button>
          </div>
        </main>

        <footer className="status">
          WebGPU accelerated · No server · RAG v2 enabled · Created by Eric Holt
        </footer>
      </div>
    </div>
  );
}
