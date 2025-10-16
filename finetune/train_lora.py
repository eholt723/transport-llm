from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

BASE="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ds = load_dataset("json", data_files={"train":"data/train.jsonl","eval":"data/eval.jsonl"})
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True); tok.pad_token = tok.eos_token

def fmt(ex):
    return f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['response']}"
def tok_map(batch): 
    enc = tok([fmt(x) for x in batch], max_length=512, truncation=True)
    enc["labels"] = enc["input_ids"].copy(); return enc
ds = ds.map(tok_map, batched=True, remove_columns=ds["train"].column_names)

bnb_args = dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, 
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", **bnb_args)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                         target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                         task_type="CAUSAL_LM"))

args = TrainingArguments(
    output_dir="out-transport", num_train_epochs=1,
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-4, bf16=True, warmup_ratio=0.05, lr_scheduler_type="cosine",
    logging_steps=20, evaluation_strategy="steps", eval_steps=200,
    save_steps=400, save_total_limit=2, report_to="none"
)
Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["eval"]).train()
model.save_pretrained("out-transport/lora_adapter"); tok.save_pretrained("out-transport")
print("Saved LoRA to out-transport/lora_adapter")
