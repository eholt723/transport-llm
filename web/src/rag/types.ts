export type Chunk = {
  id: string;
  doc_id: string;
  title: string;
  source: string;
  offset: number;
  text: string;
  meta?: {
    domain?: string; // e.g., "rail" | "auto" | "transit" | "standards"
  };
};

export type IndexJson = {
  dim: number;
  chunk_count: number;
  doc_count: number;
  model: string;
  created_utc: number;
  domains?: string[];
  chunks: Chunk[];
};

export type Retrieved = {
  chunk: Chunk;
  score: number; // post-weighting score
  raw?: number;  // raw cosine (pre-weight)
};

export type RetrievalOptions = {
  k?: number;
  domains?: string[];           // filter: only include these domains (if set)
  domainWeights?: Record<string, number>; // e.g., { rail: 1.0, auto: 0.9 }
  mmr?: { lambda?: number; fetchK?: number }; // MMR diversification
};

export type RetrievalResult = {
  query: string;
  topK: Retrieved[];
  elapsedMs: number;
  applied: {
    domains?: string[];
    domainWeights?: Record<string, number>;
    mmr?: { lambda: number; fetchK: number };
  };
};
