import type { Chunk, Retrieved } from "./types";

// Apply domain weighting to raw cosine scores.
export function applyDomainWeights(
  items: { index: number; score: number }[],
  chunks: Chunk[],
  domainWeights?: Record<string, number>
): Retrieved[] {
  return items.map(({ index, score }) => {
    const c = chunks[index];
    const d = (c.meta?.domain || "").toLowerCase();
    const w = domainWeights?.[d] ?? 1.0;
    return {
      chunk: c,
      raw: score,
      score: score * w,
    };
  });
}


export function mmr(
  _queryVec: Float32Array, 
  _emb: Float32Array,
  _dim: number,
  candidates: Retrieved[],
  lambda = 0.7,
  k = 5
): Retrieved[] {
  const selected: Retrieved[] = [];
  const pool = [...candidates];

  while (selected.length < k && pool.length > 0) {
    let bestIdx = -1;
    let bestScore = -Infinity;

    for (let p = 0; p < pool.length; p++) {
      const cand = pool[p];
      // Penalize near-duplicates by doc_id/title overlap
      let redundancy = 0;
      for (const s of selected) {
        if (s.chunk.doc_id === cand.chunk.doc_id) redundancy = Math.max(redundancy, 0.5);
        if (s.chunk.title === cand.chunk.title) redundancy = Math.max(redundancy, 0.3);
      }
      const mmrScore = lambda * cand.score - (1 - lambda) * redundancy;
      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIdx = p;
      }
    }

    selected.push(pool[bestIdx]);
    pool.splice(bestIdx, 1);
  }

  return selected;
}
