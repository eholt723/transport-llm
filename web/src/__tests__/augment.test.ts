import { describe, it, expect } from "vitest";
import { buildAugmentedPrompt } from "../rag/augment";
import type { Retrieved } from "../rag/types";

function makeRetrieved(title: string, text: string, score = 0.9): Retrieved {
  return {
    chunk: {
      id: "x#0000",
      doc_id: "x",
      title,
      source: "x.md",
      offset: 0,
      text,
    },
    score,
  };
}

describe("buildAugmentedPrompt", () => {
  it("includes the header, chunk content, and query", () => {
    const result = buildAugmentedPrompt(
      "what is rail signaling?",
      [makeRetrieved("Rail Basics", "Signaling controls train movement.")],
      { maxTokens: 500 }
    );
    expect(result).toContain("Rail Basics");
    expect(result).toContain("Signaling controls train movement.");
    expect(result).toContain("what is rail signaling?");
  });

  it("uses a custom header when provided", () => {
    const result = buildAugmentedPrompt(
      "query",
      [makeRetrieved("T", "text")],
      { maxTokens: 500, header: "Custom context:" }
    );
    expect(result).toContain("Custom context:");
  });

  it("returns a prompt even with an empty retrieved list", () => {
    const result = buildAugmentedPrompt("what is transit?", [], { maxTokens: 500 });
    expect(result).toContain("what is transit?");
  });

  it("stays within budget when content exceeds maxTokens", () => {
    const longText = "w".repeat(2000);
    const result = buildAugmentedPrompt(
      "q",
      [makeRetrieved("Doc1", longText), makeRetrieved("Doc2", longText)],
      { maxTokens: 100, charsPerToken: 4 } // 400 char budget
    );
    // hard-slice adds up to 20 chars of suffix, allow small buffer
    expect(result.length).toBeLessThanOrEqual(420);
  });

  it("includes all chunks when within budget", () => {
    const result = buildAugmentedPrompt(
      "q",
      [makeRetrieved("Doc1", "short"), makeRetrieved("Doc2", "also short")],
      { maxTokens: 500 }
    );
    expect(result).toContain("Doc1");
    expect(result).toContain("Doc2");
  });
});
