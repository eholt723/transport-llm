// web/src/rag.ts
export type Doc = { id: string; title: string; source: string; text: string };

type Tf = Map<string, number>;
type Vec = Map<string, number>;

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function tf(tokens: string[]): Tf {
  const m = new Map<string, number>();
  for (const t of tokens) m.set(t, (m.get(t) ?? 0) + 1);
  return m;
}

function normalize(v: Vec): Vec {
  let sum = 0;
  for (const [, val] of v) sum += val * val;
  const norm = Math.sqrt(sum) || 1;
  const out = new Map<string, number>();
  for (const [k, val] of v) out.set(k, val / norm);
  return out;
}

function cosine(a: Vec, b: Vec): number {
  let s = 0;
  // iterate smaller one for speed
  const [small, big] = a.size < b.size ? [a, b] : [b, a];
  for (const [k, va] of small) {
    const vb = big.get(k);
    if (vb) s += va * vb;
  }
  return s;
}

export type RagIndex = {
  idf: Map<string, number>;
  docVecs: Map<string, Vec>;
  byId: Map<string, Doc>;
};

export function buildIndex(docs: Doc[]): RagIndex {
  const byId = new Map<string, Doc>();
  const docTfs: Map<string, Tf> = new Map();
  const df = new Map<string, number>();

  for (const d of docs) {
    byId.set(d.id, d);
    const tokens = tokenize(d.text);
    const tfi = tf(tokens);
    docTfs.set(d.id, tfi);
    for (const k of tfi.keys()) df.set(k, (df.get(k) ?? 0) + 1);
  }

  const N = docs.length;
  const idf = new Map<string, number>();
  for (const [k, n] of df) idf.set(k, Math.log((1 + N) / (1 + n)) + 1);

  const docVecs = new Map<string, Vec>();
  for (const [id, tfi] of docTfs) {
    const v = new Map<string, number>();
    for (const [k, f] of tfi) v.set(k, f * (idf.get(k) ?? 0));
    docVecs.set(id, normalize(v));
  }

  return { idf, docVecs, byId };
}

function queryVec(q: string, idf: Map<string, number>): Vec {
  const tfi = tf(tokenize(q));
  const v = new Map<string, number>();
  for (const [k, f] of tfi) v.set(k, f * (idf.get(k) ?? 0));
  return normalize(v);
}

export function topK(
  q: string,
  index: RagIndex,
  k = 3
): Array<{ doc: Doc; score: number }> {
  const qv = queryVec(q, index.idf);
  const scored: Array<{ doc: Doc; score: number }> = [];
  for (const [id, dv] of index.docVecs) {
    const score = cosine(qv, dv);
    scored.push({ doc: index.byId.get(id)!, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

export function makeContext(
  q: string,
  index: RagIndex,
  k = 3,
  maxChars = 1200
): string {
  const hits = topK(q, index, k);
  let ctx = "";
  for (const { doc, score } of hits) {
    const chunk = `- [${doc.title} | ${doc.source} | score=${score.toFixed(
      3
    )}] ${doc.text}\n`;
    if (ctx.length + chunk.length > maxChars) break;
    ctx += chunk;
  }
  return ctx.trim();
}
