import { useEffect, useRef, useState } from "react";
import {
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";
import ReactMarkdown from "react-markdown";

import { retrieve } from "./rag/retriever";
import { buildAugmentedPrompt } from "./rag/augment";
import type { Retrieved } from "./rag/types";
import { DEFAULT_MODEL_ID, DOMAIN_WEIGHTS } from "./constants";

type ChatMsg = {
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Retrieved[];
};

const INITIAL_MESSAGES: ChatMsg[] = [
  {
    role: "system",
    content: `You are Transport LLM — a concise assistant focused on rail operations, automotive engineering, intelligent transit, and transport standards.
If context is provided, ground your answer in it. If the context is insufficient, say so briefly.
When you use provided context, cite sources by [title]. If a user corrects you, accept the correction politely.`,
  },
  {
    role: "assistant",
    content:
      "Welcome. I'm designed to assist with rail operations, vehicle systems, public transit engineering, and transportation standards.",
  },
];

// Maximum number of prior turns (user+assistant pairs) sent to the model.
const MAX_HISTORY_TURNS = 8;

// Minimum RAG score to inject context. Below this, skip augmentation.
const RAG_SCORE_THRESHOLD = 0.3;

export default function App() {
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [status, setStatus] = useState("Loading model…");
  const [loadProgress, setLoadProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [ragStatus, setRagStatus] = useState<"idle" | "searching">("idle");
  const [input, setInput] = useState("");
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMsg[]>(INITIAL_MESSAGES);
  const [openCitations, setOpenCitations] = useState<Set<number>>(new Set());

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const hasInitError = status.startsWith("Init error");
  const hasRuntimeError = !!runtimeError;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!busy) inputRef.current?.focus();
  }, [busy]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const e = await CreateMLCEngine(DEFAULT_MODEL_ID, {
          initProgressCallback: (p: InitProgressReport) => {
            setStatus(p.text);
            // Prefer chunk-count progress from text (e.g. "[11/108]") — more accurate
            // during cache loading when p.progress stays near 0.
            const match = p.text.match(/\[(\d+)\/(\d+)\]/);
            if (match) {
              setLoadProgress(parseInt(match[1], 10) / parseInt(match[2], 10));
            } else {
              setLoadProgress(p.progress ?? 0);
            }
          },
        });
        if (!cancelled) {
          setEngine(e);
          setStatus("Ready");
        }
      } catch (err: unknown) {
        const msg = `Init error: ${err instanceof Error ? err.message : String(err)}`;
        setStatus(msg);
        setMessages((m) => [...m, { role: "assistant", content: msg }]);
      }
    })();

    return () => { cancelled = true; };
  }, []);

  function clearConversation() {
    setMessages(INITIAL_MESSAGES);
    setOpenCitations(new Set());
    setRuntimeError(null);
    inputRef.current?.focus();
  }

  function toggleCitations(index: number) {
    setOpenCitations((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  async function send() {
    const userRaw = input.trim();
    if (!userRaw || !engine) return;

    let augmentedUser = userRaw;
    let citations: Retrieved[] = [];

    setRagStatus("searching");
    try {
      const k = 5;
      const ret = await retrieve(userRaw, {
        k,
        domains: [],
        domainWeights: DOMAIN_WEIGHTS,
        mmr: { lambda: 0.7, fetchK: Math.max(40, k * 4) },
      });

      const bestScore = ret.topK[0]?.score ?? 0;
      if (bestScore >= RAG_SCORE_THRESHOLD) {
        citations = ret.topK;
        augmentedUser = buildAugmentedPrompt(userRaw, ret.topK, {
          maxTokens: 1200,
          charsPerToken: 4,
          header: "Context (use for citations):",
        });
      }
    } catch (e: unknown) {
      console.warn("RAG retrieval failed:", e);
    } finally {
      setRagStatus("idle");
    }

    setInput("");
    const uiNext: ChatMsg[] = [
      ...messages,
      { role: "user", content: userRaw },
      { role: "assistant", content: "", citations },
    ];
    setMessages(uiNext);
    setBusy(true);

    try {
      // Build sliding window: system + last MAX_HISTORY_TURNS turns
      const nonSystem = uiNext.filter((m) => m.role !== "system");
      const window = nonSystem.slice(-(MAX_HISTORY_TURNS * 2 + 1)).slice(0, -1);
      const apiMessages = [
        { role: "system" as const, content: INITIAL_MESSAGES[0].content },
        ...window.map((m) => ({ role: m.role as "user" | "assistant", content: m.content })),
        { role: "user" as const, content: augmentedUser },
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
    } catch (err: unknown) {
      console.error("Runtime error from WebLLM:", err);
      const friendly =
        "The model runtime encountered an error and stopped. This usually means your device or browser does not fully support WebGPU for this model, or ran out of GPU memory. Try Chrome or Edge on a desktop/laptop with a modern GPU. Mobile browsers may not support WebGPU.";
      setRuntimeError(friendly);
      setMessages((curr) => [...curr, { role: "assistant", content: friendly }]);
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
  const visibleMessages = messages.filter((m) => m.role !== "system");

  return (
    <div className="app-frame">
      <div className="app-wrap">
        <header className="header">
          <h1>Transportation LLM — Edge AI Demo</h1>
          <div className="status">{ragStatus === "searching" ? "Searching knowledge base…" : status}</div>
          {!engine && !hasInitError && (
            <div className="load-bar-track">
              <div className="load-bar-fill" style={{ width: `${Math.round(loadProgress * 100)}%` }} />
            </div>
          )}
        </header>

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
            {visibleMessages.map((m, i) => {
              const globalIdx = i;
              const hasCitations = m.role === "assistant" && m.citations && m.citations.length > 0;
              const citationsOpen = openCitations.has(globalIdx);

              return (
                <div key={i} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                  {m.role === "assistant" ? (
                    <div className="prose">
                      <ReactMarkdown>{m.content || " "}</ReactMarkdown>
                    </div>
                  ) : (
                    m.content
                  )}

                  {hasCitations && (
                    <div className="citations-wrap">
                      <button
                        className="citations-toggle"
                        onClick={() => toggleCitations(globalIdx)}
                        type="button"
                      >
                        {citationsOpen ? "Hide sources" : `Show sources (${m.citations!.length})`}
                      </button>
                      {citationsOpen && (
                        <div className="citations-list">
                          {m.citations!.map(({ chunk, score }) => (
                            <div key={chunk.id} className="citation-item">
                              <div className="citation-title">
                                {chunk.title}
                                <span className="citation-score">{score.toFixed(3)}</span>
                              </div>
                              <div className="citation-meta">{chunk.source} · chunk {chunk.offset}</div>
                              <div className="citation-text">{chunk.text}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
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
            <div className="input-actions">
              <button className="button" disabled={!canSend} onClick={send}>
                {busy ? "…" : "Send"}
              </button>
              <button
                className="button clear-btn"
                onClick={clearConversation}
                disabled={busy}
                title="Clear conversation"
                type="button"
              >
                Clear
              </button>
            </div>
          </div>
        </main>

        <footer className="status">
          WebGPU accelerated · No server · RAG v2 enabled · Created by Eric Holt
        </footer>
      </div>
    </div>
  );
}
