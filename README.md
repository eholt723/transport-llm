# Transit Through Time — Offline LLM (WebGPU)
Small fine-tuned LLM that explains world transportation history. Runs 100% in-browser via WebGPU.

## Status
- [ ] Data collected
- [ ] LoRA fine-tune
- [ ] Quantized model
- [ ] WebGPU demo live

## Quick Start
1) Put public-domain texts in `data/raw_texts/`.
2) Run `python finetune/prepare_dataset.py` → `data/train.jsonl`, `data/eval.jsonl`.
3) Run `python finetune/train_lora.py`.
4) Quantize/convert for web, drop files into `web/`, and serve via GitHub Pages.
