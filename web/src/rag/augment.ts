import type { Retrieved } from "./types";

type BudgetOpts = {
  // rough token estimate: ~4 chars/token default
  maxTokens: number;
  charsPerToken?: number;
  header?: string;
  footer?: string;
};

export function buildAugmentedPrompt(
  userQuery: string,
  retrieved: Retrieved[],
  opts: BudgetOpts
): string {
  const charsPerToken = opts.charsPerToken ?? 4;
  const maxChars = opts.maxTokens * charsPerToken;
  const header = opts.header ?? "Use the following context to answer the question. Cite sources by title.";
  const footer = opts.footer ?? "";

  const citeBlockLines: string[] = [];
  for (const { chunk, score } of retrieved) {
    citeBlockLines.push(
      `### [${chunk.title}] (score=${score.toFixed(3)})\n${chunk.text}`
    );
  }

  const base = `${header}\n\n${citeBlockLines.join("\n\n")}\n\nUser question: ${userQuery}\n${footer}`.trim();

  if (base.length <= maxChars) return base;

  // truncate bottom-up until within budget
  const lines = citeBlockLines.slice();
  while (lines.length > 1) {
    lines.pop();
    const candidate = `${header}\n\n${lines.join("\n\n")}\n\nUser question: ${userQuery}\n${footer}`.trim();
    if (candidate.length <= maxChars) return candidate;
  }

  // last resort: hard slice
  return base.slice(0, maxChars - 20) + "\n...[truncated]";
}
