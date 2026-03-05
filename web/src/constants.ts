export const DEFAULT_MODEL_ID = "Llama-3-8B-Instruct-q4f16_1-MLC";

export const DOMAIN_WEIGHTS: Record<string, number> = {
  rail: 1.0,
  auto: 0.95,
  transit: 0.95,
  standards: 1.0,
};