# train_lora.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load your JSONL (expects fields: "instruction", "response")
ds = load_dataset("json", data_files={"train":"data/train.jsonl", "eval":"data/eval.jsonl"})

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def fmt(inst, resp):
    return f"### Instruction:\n{inst}\n### Response:\n{resp}"

def tok_map(batch):
    texts = [fmt(i, r) for i, r in zip(batch["instruction"], batch["response"])]
    enc = tok(texts, max_length=512, truncation=True)
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds = ds.map(tok_map, batched=True, remove_columns=ds["train"].column_names)

# Try 4-bit (if bitsandbytes available); otherwise fall back to standard loading
bnb_kwargs = dict(load_in_4bit=True,
                  bnb_4bit_use_double_quant=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16)

try:
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", **bnb_kwargs)
    model = prepare_model_for_kbit_training(model)
    fourbit = True
except Exception as e:
    print("⚠️ 4-bit load failed, falling back to standard weights:", e)
    model = AutoModelForCausalLM.from_pretrained(BASE)
    fourbit = False

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# bf16 only if supported; else try fp16 on GPU; else CPU
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
use_fp16 = torch.cuda.is_available() and not use_bf16

args = TrainingArguments(
    output_dir="out-transport",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=400,
    save_total_limit=2,
    report_to="none",
    bf16=use_bf16,
    fp16=use_fp16
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["eval"])
trainer.train()

model.save_pretrained("out-transport/lora_adapter")
tok.save_pretrained("out-transport")
print("✅ Saved LoRA to out-transport/lora_adapter")
