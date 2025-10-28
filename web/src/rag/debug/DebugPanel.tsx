import React, { useEffect, useState } from "react";
import { retrieve, getIndexInfo } from "../retriever";
import { Citations } from "../components/Citations";

export default function DebugPanel() {
  const [info, setInfo] = useState<any>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getIndexInfo().then(setInfo).catch(console.error);
  }, []);

  async function onSearch() {
    setBusy(true);
    try {
      const res = await retrieve(q, 5);
      setHits(res);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border rounded-2xl p-4">
      <div className="text-sm opacity-70">RAG index</div>
      {info && (
        <div className="text-sm">
          dim={info.dim}, chunks={info.chunks}, docs={info.docs}, model={info.model}
        </div>
      )}
      <div className="mt-3 flex gap-2">
        <input
          className="border rounded-xl p-2 flex-1"
          placeholder="Try a question about your corpus…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <button className="border rounded-xl px-3" onClick={onSearch} disabled={busy || !q}>
          {busy ? "Searching…" : "Search"}
        </button>
      </div>
      {hits && (
        <>
          <div className="text-sm mt-3 opacity-70">
            {hits.topK.length} results in {hits.elapsedMs.toFixed(1)} ms
          </div>
          <Citations items={hits.topK} />
        </>
      )}
    </div>
  );
}
