import { useEffect, useRef, useState } from "react";
import {
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import docs from "./data/docs.json";
import { buildIndex, makeContext } from "./rag";

type ChatMsg = { role: "user" | "assistant" | "system"; content: string };

const catalogIds = prebuiltAppConfig.model_list.map((m) => m.model_id);
const DEFAULT_MODEL_ID =
  catalogIds.find(
    (id) =>
      /Llama-3.*8B.*Instruct/i.test(id) && /q4f16_1/i.test(id) && /-MLC$/i.test(id)
  ) ?? catalogIds[0] ?? "Llama-3-8B-Instruct-q4f16_1-MLC";

const USE_RAG_DEFAULT = true;

export default function App() {
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [status, setStatus] = useState("Loading model…");
  const [busy, setBusy] = useState(true);
  const [input, setInput] = useState("");
  const [modelId, setModelId] = useState<string>(DEFAULT_MODEL_ID);
  const [useRag, setUseRag] = useState<boolean>(USE_RAG_DEFAULT);
  const [messages, setMessages] = useState<ChatMsg[]>([
    {
      role: "system",
      content:
        "You are a concise assistant for the Transport LLM — Edge AI Demo.",
    },
    { role: "assistant", content: "Model is loading… first run may take a bit." },
  ]);

  // Build the RAG index once
  const ragIndexRef = useRef(buildIndex(docs as any));

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [busy]);

  useEffect(() => {
    let cancelled = false;
    async function init(model: string) {
      setBusy(true);
      setStatus(`Loading ${model}…`);
      try {
        const e = await CreateMLCEngine(model, {
          initProgressCallback: (p: InitProgressReport) => setStatus(p.text),
        });
        if (!cancelled) {
          setEngine(e);
          setStatus("Ready");
          setMessages((m) => [
            m[0],
            { role: "assistant", content: `Ready. Using model: ${model}` },
          ]);
        }
      } catch (err: any) {
        const msg = `Init error: ${err?.message ?? String(err)}`;
        setStatus(msg);
        setMessages((m) => [...m, { role: "assistant", content: msg }]);
        setEngine(null);
      } finally {
        if (!cancelled) setBusy(false);
      }
    }
    init(modelId);
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  async function send() {
    const userRaw = input.trim();
    if (!userRaw || !engine) return;

    // Build context if RAG is on
    let userWithContext = userRaw;
    if (useRag) {
      const ctx = makeContext(userRaw, ragIndexRef.current, 3, 1200);
      if (ctx) {
        userWithContext =
          `Use the transportation context below to answer the question. ` +
          `If the context is insufficient, say so and answer best-effort.\n\n` +
          `### Context\n${ctx}\n\n` +
          `### Question\n${userRaw}`;
      }
    }

    setInput("");
    const next = [
      ...messages,
      { role: "user", content: userWithContext } as ChatMsg,
      { role: "assistant", content: "" } as ChatMsg,
    ];
    setMessages(next);
    setBusy(true);

    try {
      // IMPORTANT: drop the trailing assistant placeholder when calling the API
      const apiMessages = next.filter((m) => m.role !== "system").slice(0, -1);

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

          {/* Model picker (dev-time). We’ll remove at polish time. */}
          <div style={{ marginTop: 8, display: "flex", gap: 12, justifyContent: "center" }}>
            <label style={{ fontSize: 12, color: "var(--muted)" }}>
              Model:&nbsp;
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                disabled={busy}
                style={{
                  background: "#0f141a",
                  color: "var(--text)",
                  border: "1px solid #334150",
                  borderRadius: 8,
                  padding: "4px 8px",
                }}
              >
                {catalogIds.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ fontSize: 12, color: "var(--muted)", display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
                disabled={busy}
              />
              Use RAG context
            </label>
          </div>
        </header>

        <main className="card chat">
          <div className="chat-scroll" ref={scrollRef}>
            {messages
              .filter((m) => m.role !== "system")
              .map((m, i) => (
                <div
                  key={i}
                  className={`msg ${m.role === "user" ? "user" : "assistant"}`}
                >
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
            <button
              className="button"
              disabled={!engine || busy || !input.trim()}
              onClick={send}
            >
              {busy ? "…" : "Send"}
            </button>
          </div>
        </main>

        <footer className="status">
          Fully local via WebGPU &nbsp;|&nbsp; No server, no API keys &nbsp;|&nbsp; RAG v1 (TF-IDF)
        </footer>
      </div>
    </div>
  );
}
