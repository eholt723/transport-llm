import { describe, it, expect } from "vitest";
import { applyDomainWeights, diversify } from "../rag/scorer";
import type { Chunk, Retrieved } from "../rag/types";

function makeChunk(overrides: Partial<Chunk> = {}): Chunk {
  return {
    id: "a#0000",
    doc_id: "a",
    title: "Title A",
    source: "a.md",
    offset: 0,
    text: "Some text.",
    ...overrides,
  };
}

function makeCandidate(doc_id: string, score: number, title = "T"): Retrieved {
  return { chunk: makeChunk({ doc_id, id: `${doc_id}#0000`, title }), score };
}

const dummy = new Float32Array(2);

describe("applyDomainWeights", () => {
  const chunks: Chunk[] = [
    makeChunk({ meta: { domain: "rail" } }),
    makeChunk({ meta: { domain: "auto" } }),
    makeChunk({ meta: { domain: "unknown" } }),
  ];

  it("applies matching domain weight to score", () => {
    const result = applyDomainWeights([{ index: 0, score: 1.0 }], chunks, { rail: 1.2 });
    expect(result[0].score).toBeCloseTo(1.2);
    expect(result[0].raw).toBeCloseTo(1.0);
  });

  it("defaults to weight 1.0 for unrecognized domain", () => {
    const result = applyDomainWeights([{ index: 2, score: 0.8 }], chunks, { rail: 1.2 });
    expect(result[0].score).toBeCloseTo(0.8);
  });

  it("defaults to weight 1.0 when no weights provided", () => {
    const result = applyDomainWeights([{ index: 0, score: 0.9 }], chunks);
    expect(result[0].score).toBeCloseTo(0.9);
  });
});

describe("diversify", () => {
  it("returns exactly k results when pool is large enough", () => {
    const candidates = [
      makeCandidate("a", 0.9),
      makeCandidate("b", 0.8),
      makeCandidate("c", 0.7),
      makeCandidate("d", 0.6),
    ];
    expect(diversify(dummy, dummy, 2, candidates, 0.7, 3)).toHaveLength(3);
  });

  it("returns fewer than k if pool is smaller", () => {
    const candidates = [makeCandidate("a", 0.9)];
    expect(diversify(dummy, dummy, 2, candidates, 0.7, 3)).toHaveLength(1);
  });

  it("returns empty array for empty candidates", () => {
    expect(diversify(dummy, dummy, 2, [], 0.7, 3)).toHaveLength(0);
  });

  it("penalizes chunks from the same doc_id", () => {
    const candidates = [
      { chunk: makeChunk({ doc_id: "a", id: "a#0000", title: "Doc A" }), score: 0.9 },
      { chunk: makeChunk({ doc_id: "a", id: "a#0001", title: "Doc A" }), score: 0.85 },
      { chunk: makeChunk({ doc_id: "b", id: "b#0000", title: "Doc B" }), score: 0.7 },
    ];
    const result = diversify(dummy, dummy, 2, candidates, 0.7, 3);
    // "b" should be selected before the second "a" chunk
    const ids = result.map((r) => r.chunk.doc_id);
    expect(ids.indexOf("b")).toBeLessThan(ids.lastIndexOf("a"));
  });
});
