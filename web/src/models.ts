export type WebModel = {
  id: string;
  label: string;
  ggufUrl: string;
  contextLength: number;
  quant: string;
};

// Primary = your trained model (default)
export const PROD: WebModel = {
  id: "lora-merged-q4km",
  label: "LoRA-Merged (Q4_K_M)",
  ggufUrl:
    "https://huggingface.co/eholt723/transport-llm-gguf/resolve/main/merged-Q4_K_M.gguf",
  contextLength: 2048,
  quant: "Q4_K_M",
};

// Hidden fallback — for now, point to PROD (you can replace this URL later)
export const BASE: WebModel = {
  id: "base-fallback",
  label: "Base Fallback (Q4_K_M)",
  ggufUrl:
    "https://huggingface.co/eholt723/transport-llm-gguf/resolve/main/merged-Q4_K_M.gguf",
  contextLength: 2048,
  quant: "Q4_K_M",
};
