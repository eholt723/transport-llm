import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_DIR = "out-transport/lora_adapter"
OUT_DIR = "out-transport/merged"

os.makedirs(OUT_DIR, exist_ok=True)
print("[merge] Loading base..."); base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="cpu")
print("[merge] Loading adapter..."); model = PeftModel.from_pretrained(base, ADAPTER_DIR)
print("[merge] Merging..."); model = model.merge_and_unload()
print("[merge] Saving..."); model.save_pretrained(OUT_DIR)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.save_pretrained(OUT_DIR)
print("[merge] ✅ Saved to", OUT_DIR)
