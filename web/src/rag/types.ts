export type Chunk = {
  id: string;
  doc_id: string;
  title: string;
  source: string;   // relative path or label
  offset: number;
  text: string;
};

export type IndexJson = {
  dim: number;
  chunk_count: number;
  doc_count: number;
  model: string;
  created_utc: number;
  chunks: Chunk[];
};

export type Retrieved = {
  chunk: Chunk;
  score: number; // cosine sim
};

export type RetrievalResult = {
  query: string;
  topK: Retrieved[];
  elapsedMs: number;
};
