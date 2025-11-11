import type { IndexJson, RetrievalOptions, RetrievalResult } from "./types";
import { l2Normalize, topKCosine } from "./similarity";
import { applyDomainWeights, mmr } from "./scorer";

let _index: IndexJson | null = null;
let _emb: Float32Array | null = null;
let _pipe: any | null = null;

async function ensureIndex(): Promise<void> {
  if (_index && _emb) return;
  const [idxResp, embResp] = await Promise.all([
    fetch("/rag/index.json"),
    fetch("/rag/embeddings.f32"),
  ]);
  if (!idxResp.ok || !embResp.ok) {
    throw new Error("Failed to load RAG index files from /rag/");
  }
  _index = (await idxResp.json()) as IndexJson;
  const buf = await embResp.arrayBuffer();
  _emb = new Float32Array(buf);
}

async function ensureEmbedder(): Promise<void> {
  if (_pipe) return;
  const { pipeline } = await import("@xenova/transformers");
  _pipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });
}

async function embedQuery(text: string): Promise<Float32Array> {
  await ensureEmbedder();
  const output = await _pipe(text, { pooling: "mean", normalize: true });
  const arr = Array.isArray(output.data) ? new Float32Array(output.data) : (output.data as Float32Array);
  return arr;
}

export async function retrieve(
  query: string,
  kOrOpts: number | RetrievalOptions = 5
): Promise<RetrievalResult> {
  await ensureIndex();
  if (!_index || !_emb) throw new Error("Index not loaded");

  const opts: RetrievalOptions = typeof kOrOpts === "number" ? { k: kOrOpts } : (kOrOpts || {});
  const k = opts.k ?? 5;

  const t0 = performance.now();
  const q = await embedQuery(query);
  l2Normalize(q);

  // wider candidate pool
  const fetchK = Math.max(k * 4, opts.mmr?.fetchK ?? 40);
  const rawHits = topKCosine(q, _emb!, _index!.dim, fetchK);

  // Optional domain filtering
  let filtered = rawHits;
  if (opts.domains && opts.domains.length > 0) {
    const allow = new Set(opts.domains.map((d) => d.toLowerCase()));
    filtered = rawHits.filter(({ index }) => {
      const dom = (_index!.chunks[index].meta?.domain || "").toLowerCase();
      return allow.has(dom);
    });
  }

  // Apply domain weights
  let weighted = applyDomainWeights(filtered, _index!.chunks, opts.domainWeights);

  // Diversify with MMR heuristic
  if (opts.mmr) {
    const lam = opts.mmr.lambda ?? 0.7;
    const kk = k;
    weighted = mmr(q, _emb!, _index!.dim, weighted, lam, kk);
  } else {
    // Fallback
    weighted = weighted.sort((a, b) => b.score - a.score).slice(0, k);
  }

  const elapsedMs = performance.now() - t0;
  return {
    query,
    topK: weighted,
    elapsedMs,
    applied: {
      domains: opts.domains,
      domainWeights: opts.domainWeights,
      mmr: opts.mmr ? { lambda: opts.mmr.lambda ?? 0.7, fetchK } : undefined,
    },
  };
}

export async function getIndexInfo() {
  await ensureIndex();
  return {
    dim: _index!.dim,
    chunks: _index!.chunk_count,
    docs: _index!.doc_count,
    model: _index!.model,
    created_utc: _index!.created_utc,
    domains: _index!.domains ?? [],
  };
}
