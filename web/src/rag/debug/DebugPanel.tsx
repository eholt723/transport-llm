import { useEffect, useState } from "react";
import { retrieve, getIndexInfo } from "../retriever";
import { Citations } from "../components/Citations";
import Filters from "../components/Filters";

export default function DebugPanel() {
  const [info, setInfo] = useState<any>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [domains, setDomains] = useState<string[]>([]); 

  useEffect(() => {
    getIndexInfo().then(setInfo).catch(console.error);
  }, []);

  async function onSearch() {
    setBusy(true);
    try {
      const res = await retrieve(q, {
        k: 5,
        domains: domains.length ? domains : undefined,
        domainWeights: { rail: 1.0, auto: 0.95, transit: 0.95, standards: 1.0 },
        mmr: { lambda: 0.7, fetchK: 40 },
      });
      setHits(res);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="border rounded-2xl p-4">
      <div className="text-sm opacity-70">RAG index</div>
      {info && (
        <div className="text-sm mb-2">
          dim={info.dim}, chunks={info.chunks}, docs={info.docs}, model={info.model}
        </div>
      )}
      <div className="mb-3">
        <Filters
          available={info?.domains || []}
          selected={domains}
          onChange={setDomains}
        />
      </div>
      <div className="mt-1 flex gap-2">
        <input
          className="border rounded-xl p-2 flex-1"
          placeholder="Try a question…"
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
            {hits.applied.domains?.length ? ` • domains=${hits.applied.domains.join(",")}` : ""}
          </div>
          <Citations items={hits.topK} />
        </>
      )}
    </div>
  );
}
