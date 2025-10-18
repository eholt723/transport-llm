# finetune/quick_infer.py
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "out-transport/lora_adapter"

def load():
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def gen(model, tok, q, max_new_tokens=30):
    prompt = f"### Instruction:\n{q}\n\n### Response:\n"
    x = tok(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.time() - t0
    text = tok.decode(y[0], skip_special_tokens=True)
    print(text)
    print(f"\n---\nGenerated {max_new_tokens} tokens in {dt:.2f}s (~{max_new_tokens/max(dt,1e-6):.1f} tok/s)")

if __name__ == "__main__":
    tok, model = load()
    gen(model, tok, "Define headway in transit.", max_new_tokens=24)
    gen(model, tok, "List 3 benefits of bus rapid transit.", max_new_tokens=60)
