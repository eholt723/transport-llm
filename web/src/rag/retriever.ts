import type { IndexJson, RetrievalResult } from "./types";
import { l2Normalize, topKCosine } from "./similarity";

// Lazy-loaded global state
let _index: IndexJson | null = null;
let _emb: Float32Array | null = null;

// On-demand transformer pipeline
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
  // @xenova/transformers loads models from CDN and caches locally
  const { pipeline } = await import("@xenova/transformers");
  _pipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
    quantized: true,
  });
}

async function embedQuery(text: string): Promise<Float32Array> {
  await ensureEmbedder();
  // mean pooling over last hidden states
  const output = await _pipe(text, { pooling: "mean", normalize: true });
  // output is Float32Array or Tensor-like
  const arr = Array.isArray(output.data) ? new Float32Array(output.data) : (output.data as Float32Array);
  // already normalized by the pipeline (normalize: true), but keep API consistent
  return arr;
}

export async function retrieve(
  query: string,
  k = 5
): Promise<RetrievalResult> {
  await ensureIndex();
  if (!_index || !_emb) throw new Error("Index not loaded");
  const t0 = performance.now();
  const q = await embedQuery(query); // dim matches _index.dim
  // safety: normalize in case upstream changes
  // Not strictly necessary if normalize:true worked, but harmless
  l2Normalize(q);
  const hits = topKCosine(q, _emb!, _index!.dim, k);
  const topK = hits.map(({ index, score }) => ({
    chunk: _index!.chunks[index],
    score,
  }));
  const elapsedMs = performance.now() - t0;
  return { query, topK, elapsedMs };
}

export async function getIndexInfo() {
  await ensureIndex();
  return {
    dim: _index!.dim,
    chunks: _index!.chunk_count,
    docs: _index!.doc_count,
    model: _index!.model,
    created_utc: _index!.created_utc,
  };
}
