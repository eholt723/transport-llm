import { describe, it, expect } from "vitest";
import { l2Normalize, cosineSim, topKCosine } from "../rag/similarity";

describe("l2Normalize", () => {
  it("normalizes a non-zero vector to unit length", () => {
    const v = new Float32Array([3, 4]);
    l2Normalize(v);
    expect(v[0]).toBeCloseTo(0.6);
    expect(v[1]).toBeCloseTo(0.8);
  });

  it("leaves a zero vector unchanged", () => {
    const v = new Float32Array([0, 0, 0]);
    l2Normalize(v);
    expect(Array.from(v)).toEqual([0, 0, 0]);
  });
});

describe("cosineSim", () => {
  it("returns 1 for identical unit vectors", () => {
    const v = new Float32Array([0.6, 0.8]);
    expect(cosineSim(v, v)).toBeCloseTo(1);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineSim(a, b)).toBeCloseTo(0);
  });
});

describe("topKCosine", () => {
  // matrix rows: [1,0], [0,1], [0.6,0.8]
  const matrix = new Float32Array([1, 0, 0, 1, 0.6, 0.8]);

  it("returns at most k results", () => {
    const query = new Float32Array([1, 0]);
    expect(topKCosine(query, matrix, 2, 2)).toHaveLength(2);
  });

  it("returns results sorted by score descending", () => {
    const query = new Float32Array([1, 0]);
    const hits = topKCosine(query, matrix, 2, 3);
    for (let i = 1; i < hits.length; i++) {
      expect(hits[i - 1].score).toBeGreaterThanOrEqual(hits[i].score);
    }
  });

  it("handles k larger than number of vectors", () => {
    const query = new Float32Array([1, 0]);
    const single = new Float32Array([1, 0]);
    expect(topKCosine(query, single, 2, 10)).toHaveLength(1);
  });

  it("returns the best matching vector first", () => {
    const query = new Float32Array([1, 0]);
    const hits = topKCosine(query, matrix, 2, 3);
    expect(hits[0].index).toBe(0); // [1,0] is the perfect match
  });
});
