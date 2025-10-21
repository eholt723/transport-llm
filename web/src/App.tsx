import { useEffect, useRef, useState } from "react";
import {
  CreateMLCEngine,
  prebuiltAppConfig,
  type InitProgressReport,
} from "@mlc-ai/web-llm";

type ChatMsg = { role: "user" | "assistant" | "system"; content: string };

// --- helpers: choose default vs hidden fallback (?model=base) ---
function pickModelId(): string {
  const ids: string[] =
    prebuiltAppConfig?.model_list?.map((m: any) => m.model_id) ?? [];
  if (!ids.length) throw new Error("No prebuilt models found.");

  const param = new URLSearchParams(window.location.search).get("model");
  if (param === "base" && ids.length > 1) {
    // hidden fallback: choose a different model id (use the last one)
    return ids[ids.length - 1];
  }
  // default: first model id
  return ids[0];
}

export default function App() {
  const [status, setStatus] = useState("Loading model…");
  const [engine, setEngine] = useState<any>(null);
  const [busy, setBusy] = useState(true);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [modelId, setModelId] = useState<string | null>(null);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight });
  }, [messages]);

  // Keep cursor focused
  useEffect(() => {
    inputRef.current?.focus();
  }, [engine, busy, messages]);

  // Focus on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Auto-load model with hidden fallback support
  useEffect(() => {
    (async () => {
      try {
        setBusy(true);
        const onProgress = (p: InitProgressReport) =>
          setStatus(p.text || "Loading…");

        const chosen = pickModelId();
        setModelId(chosen);

        const eng = await CreateMLCEngine(chosen, {
          initProgressCallback: onProgress,
        });

        setEngine(eng);
        setStatus(`✅ Model “${chosen}” ready!`);
      } catch (err: any) {
        console.error(err);
        setStatus("Failed to load model: " + (err?.message ?? String(err)));

        // If default failed and user didn't explicitly request base, auto-redirect to fallback
        const params = new URLSearchParams(location.search);
        const requested = params.get("model");
        if (requested !== "base") {
          params.set("model", "base");
          location.search = params.toString(); // reload with fallback
        }
      } finally {
        setBusy(false);
      }
    })();
  }, []);

  async function handleSend() {
    if (!engine) {
      setStatus("Model not loaded yet.");
      return;
    }
    const prompt = input.trim();
    if (!prompt) return;

    setMessages((m) => [...m, { role: "user", content: prompt }]);
    setInput("");
    setBusy(true);
    setStatus("Generating…");

    const idx = messages.length + 1;
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    try {
      const stream = await engine.chat.completions.create({
        messages: [
          { role: "system", content: "You are a concise, helpful assistant." },
          ...messages,
          { role: "user", content: prompt },
        ],
        stream: true,
      });

      let full = "";
      for await (const chunk of stream) {
        const delta = chunk?.choices?.[0]?.delta?.content ?? "";
        if (delta) {
          full += delta;
          setMessages((m) => {
            const copy = m.slice();
            copy[idx] = { role: "assistant", content: full };
            return copy;
          });
        }
      }

      setStatus("Done.");
    } catch (err: any) {
      console.error(err);
      setStatus("Generation failed: " + (err?.message ?? String(err)));
      setMessages((m) => {
        const copy = m.slice();
        copy[idx] = {
          role: "assistant",
          content: "(Error) " + (err?.message ?? String(err)),
        };
        return copy;
      });
    } finally {
      setBusy(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!busy && engine && input.trim()) handleSend();
    }
  }

  // ── Theme & Layout ──
  const bg = "#0f172a";
  const panel = "#0b1225";
  const panelBorder = "rgba(255,255,255,0.06)";
  const text = "rgba(255,255,255,0.92)";
  const subtext = "rgba(255,255,255,0.6)";
  const inputBg = "#0d1530";
  const btn = "#4f6bff";
  const fontFamily = "system-ui, Segoe UI, Roboto, Arial, sans-serif";
  const assistantBg = "rgba(79,107,255,0.12)";
  const userBg = "#1e293b";

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        overflow: "hidden",
        background: bg,
        color: text,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily,
        padding: "6px 16px 24px 16px",
        boxSizing: "border-box",
        border: "5px solid rgba(255, 255, 255, 0.75)",
        boxShadow: "0 0 30px rgba(255, 255, 255, 0.05)",
      }}
    >
      <div style={{ width: "min(960px, 100%)" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 16 }}>
          <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700 }}>
            Transportation LLM — Edge AI Demo
          </h1>
          <div style={{ marginTop: 8, color: subtext, fontSize: 13 }}>
            Runs 100% in your browser via WebGPU
          </div>
          <div style={{ marginTop: 8, color: subtext, fontSize: 13 }}>
            {status}
            {modelId && (
              <>
                {" "}
                <span style={{ opacity: 0.7 }}>
                  {new URLSearchParams(window.location.search).get("model") ===
                  "base"
                    ? `(fallback: ${modelId})`
                    : `(default: ${modelId})`}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Chat panel */}
        <div
          style={{
            background: panel,
            border: `1px solid ${panelBorder}`,
            borderRadius: 20,
            height: 540,
            padding: "20px 24px",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            boxShadow: "0 10px 40px rgba(0,0,0,0.4)",
          }}
        >
          <div
            ref={scrollerRef}
            style={{
              flex: 1,
              overflowY: "auto",
              paddingRight: 4,
              scrollBehavior: "smooth",
            }}
          >
            {messages.length === 0 && (
              <div
                style={{
                  color: subtext,
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 15,
                }}
              >
                Start the conversation…
              </div>
            )}

            {messages.map((m, i) => {
              const isUser = m.role === "user";
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    justifyContent: isUser ? "flex-end" : "flex-start",
                    marginBottom: 16,
                  }}
                >
                  <div
                    style={{
                      maxWidth: "90%",
                      background: isUser ? userBg : assistantBg,
                      border: `1px solid ${panelBorder}`,
                      padding: "14px 18px",
                      borderRadius: 16,
                      whiteSpace: "pre-wrap",
                      lineHeight: 1.55,
                      fontSize: 15,
                      color: isUser ? text : "rgba(255,255,255,0.95)",
                      boxShadow: isUser
                        ? "0 2px 6px rgba(0,0,0,0.25)"
                        : "0 2px 10px rgba(79,107,255,0.2)",
                    }}
                  >
                    {m.content}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Input row */}
          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
            <textarea
              ref={inputRef}
              placeholder="Say hi…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              style={{
                flex: 1,
                background: inputBg,
                color: text,
                border: `1px solid ${panelBorder}`,
                outline: "none",
                borderRadius: 12,
                padding: "12px 14px",
                resize: "none",
                fontSize: 15,
                fontFamily,
              }}
              disabled={!engine || busy}
            />
            <button
              onClick={handleSend}
              disabled={!engine || busy || !input.trim()}
              style={{
                minWidth: 84,
                padding: "12px 18px",
                background: !engine || busy || !input.trim() ? "#334155" : btn,
                color: "#fff",
                border: "none",
                borderRadius: 12,
                fontWeight: 700,
                cursor:
                  !engine || busy || !input.trim() ? "not-allowed" : "pointer",
                fontFamily,
                fontSize: 15,
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
