# finetune/train_lora.py
import os, json, sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ---------------------------
# Config (safe, beginner-friendly)
# ---------------------------
BASE = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TRAIN_PATH = "data/train.jsonl"
EVAL_PATH  = "data/eval.jsonl"
OUT_DIR    = "out-transport"

MAX_LEN = 256           # keep small for speed/VRAM
EPOCHS  = 2             # start with 1 for a quick run
LR      = 2e-4
BATCH_PER_DEVICE = 1    # small to reduce OOM risk
GRAD_ACCUM = 16          # effective batch = 2 * 8 = 16 examples
LOG_STEPS = 20

# ---------------------------
# Helpers / Diagnostics
# ---------------------------
def log(s): print(f"[train_lora] {s}")

def ensure_sample_data():
    """Create tiny sample data if files are missing (so beginners can run immediately)."""
    Path("data").mkdir(exist_ok=True)
    if not Path(TRAIN_PATH).exists() or not Path(EVAL_PATH).exists():
        log("No data found. Creating tiny sample dataset...")
        samples = [
            {"instruction": "What is a bus?", "response": "A road vehicle designed to carry many passengers."},
            {"instruction": "Name a benefit of subways.", "response": "High capacity transit with reliable travel times."},
            {"instruction": "Why are bike lanes useful?", "response": "They improve safety and encourage cycling."},
            {"instruction": "Define headway in transit.", "response": "Time between vehicles on the same route."},
        ]
        with open(TRAIN_PATH, "w", encoding="utf-8") as f:
            for ex in samples: f.write(json.dumps(ex) + "\n")
        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            for ex in samples[:2]: f.write(json.dumps(ex) + "\n")
        log("Wrote tiny sample train/eval JSONL files.")

def safe_dtype():
    if torch.cuda.is_available():
        cc_major = torch.cuda.get_device_capability(0)[0]
        # BF16 is okay on Ampere+ (cc 8.0+) – but we’ll just return 'auto' and let Trainer flags handle it
        return torch.bfloat16 if cc_major >= 8 else torch.float16
    return torch.float32  # CPU

# ---------------------------
# Load dataset
# ---------------------------
def load_transport_dataset():
    # Expects fields: "instruction", "response"
    ds = load_dataset("json", data_files={"train": TRAIN_PATH, "eval": EVAL_PATH})
    # Quick schema check
    for split in ["train", "eval"]:
        cols = set(ds[split].column_names)
        need = {"instruction", "response"}
        if not need.issubset(cols):
            raise ValueError(f"{split} split must have columns {need}, but got {cols}")
    return ds

# ---------------------------
# Tokenization
# ---------------------------
def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    # Ensure padding token exists
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def format_example(inst, resp):
    return f"### Instruction:\n{inst}\n\n### Response:\n{resp}"

def tokenize_dataset(ds, tok):
    def tok_map(batch):
        texts = [format_example(i, r) for i, r in zip(batch["instruction"], batch["response"])]
        enc = tok(
            texts,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",   # 👈 force uniform length
            return_attention_mask=True,
        )
        # labels must match input_ids shape
        enc["labels"] = enc["input_ids"].copy()
        return enc

    cols = ds["train"].column_names
    ds_tok = ds.map(tok_map, batched=True, remove_columns=cols)
    return ds_tok


# ---------------------------
# Model loading (no-crash even if bitsandbytes missing)
# ---------------------------
def try_load_model_4bit():
    """Try 4-bit with bitsandbytes; return (model, used_fourbit: bool)."""
    try:
        bnb_kwargs = dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", **bnb_kwargs)
        return model, True
    except Exception as e:
        log(f"4-bit load failed or unavailable. Continuing with standard weights. Details: {e}")
        # Fallback: regular weights, pick safe dtype on GPU
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = safe_dtype()
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(BASE, **kwargs)
        return model, False

# ---------------------------
# Main
# ---------------------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    ensure_sample_data()
    ds = load_transport_dataset()
    tok = build_tokenizer()

    ds_tok = tokenize_dataset(ds, tok)

    # Data collator handles dynamic padding correctly for Causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    model, fourbit = try_load_model_4bit()
    log(f"Loaded base model: {BASE} (4-bit: {fourbit})")

    # Make sure pad token id is set on the model config (avoids warnings)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    # LoRA config
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # good for LLaMA-style models
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Dtypes: let Trainer flags control AMP; we’ll compute them here safely
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # If you get OOM, try lowering per_device_train_batch_size or MAX_LEN
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=LOG_STEPS,        
        save_steps=400,
        save_total_limit=2,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",     # safe default everywhere
        gradient_checkpointing=False,  # set True if you want lower VRAM, slower
        dataloader_pin_memory=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device} | bf16={use_bf16} fp16={use_fp16}")
    log(f"Train size: {len(ds_tok['train'])} | Eval size: {len(ds_tok['eval'])}")

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["eval"],
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    adapter_dir = os.path.join(OUT_DIR, "lora_adapter")
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(OUT_DIR)

    log(f"✅ Saved LoRA adapter to {adapter_dir}")
    log("Done.")

if __name__ == "__main__":
    main()
