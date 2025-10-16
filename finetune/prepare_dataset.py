import json, random, re
from pathlib import Path

RAW_DIR = Path("data/raw_texts")
OUT_TRAIN = Path("data/train.jsonl")
OUT_EVAL = Path("data/eval.jsonl")

def clean(t): 
    t = re.sub(r"\s+", " ", t.strip())
    return t

def to_pairs(text):
    # Naive seed: split into short “facts” → Q/A pairs you can improve later
    facts = [s.strip() for s in re.split(r"[.?!]\s+", text) if len(s.split())>6]
    pairs = []
    for f in facts:
        q = f"Explain this transportation fact in plain English: {f[:160]}"
        a = f
        pairs.append({"instruction": q, "response": a})
    return pairs

examples = []
for p in RAW_DIR.glob("*.txt"):
    txt = clean(p.read_text(encoding="utf-8", errors="ignore"))
    examples += to_pairs(txt)

random.shuffle(examples)
k = max(100, int(0.1 * len(examples)))
eval_set, train_set = examples[:k], examples[k:]

OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
with OUT_TRAIN.open("w", encoding="utf-8") as f:
    for ex in train_set: f.write(json.dumps(ex)+"\n")
with OUT_EVAL.open("w", encoding="utf-8") as f:
    for ex in eval_set: f.write(json.dumps(ex)+"\n")

print(f"train={len(train_set)} eval={len(eval_set)}")
