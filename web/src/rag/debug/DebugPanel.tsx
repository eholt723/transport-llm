import { useEffect, useState } from "react";
import { retrieve, getIndexInfo, type IndexInfo } from "../retriever";
import type { RetrievalResult } from "../types";
import { Citations } from "../components/Citations";
import Filters from "../components/Filters";
import { DOMAIN_WEIGHTS } from "../../constants";

export default function DebugPanel() {
  const [info, setInfo] = useState<IndexInfo | null>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<RetrievalResult | null>(null);
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
        domainWeights: DOMAIN_WEIGHTS,
        mmr: { lambda: 0.7, fetchK: 40 },
      });
      setHits(res);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="debug-panel">
      <div className="debug-label">RAG index</div>
      {info && (
        <div className="debug-info">
          dim={info.dim}, chunks={info.chunks}, docs={info.docs}, model={info.model}
        </div>
      )}
      <div className="debug-filters">
        <Filters
          available={info?.domains || []}
          selected={domains}
          onChange={setDomains}
        />
      </div>
      <div className="debug-search-row">
        <input
          className="debug-input"
          placeholder="Try a question…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <button className="button" onClick={onSearch} disabled={busy || !q}>
          {busy ? "Searching…" : "Search"}
        </button>
      </div>
      {hits && (
        <>
          <div className="debug-results-meta">
            {hits.topK.length} results in {hits.elapsedMs.toFixed(1)} ms
            {hits.applied.domains?.length ? ` · domains=${hits.applied.domains.join(",")}` : ""}
          </div>
          <Citations items={hits.topK} />
        </>
      )}
    </div>
  );
}
