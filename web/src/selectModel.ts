import { PROD, BASE, WebModel } from "./models";

export function selectModelFromQuery(): WebModel {
  const p = new URLSearchParams(window.location.search).get("model");
  if (p === "base") return BASE;     // hidden: ...?model=base
  return PROD;                       // default: your trained model
}

// Optional: versioned cache key (helps invalidate old caches)
export function modelCacheKey(m: WebModel) {
  const v = "v1"; // bump when you upload a new .gguf
  return `transport-llm@${m.id}@${m.quant}@${v}`;
}
