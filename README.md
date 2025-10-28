#  Transport LLM — Edge AI Demo

### Problem Framing

Modern language models are powerful but expensive, cloud-dependent, and often trained on data that isn’t specialized for real-world domains.  
The challenge: **Can we design, train, and deploy a small domain-specific LLM entirely on a personal machine**, using open-source tools, that performs useful tasks without cloud APIs?

This project explores that question using the **transportation sector** as a case study — covering history, engineering standards, terminology, and logistics.  
The goal is to show how a single developer can go end-to-end:
1. Curate a local corpus (e.g., manuals, glossaries, standards).
2. Train or fine-tune a compact model locally or in Colab.
3. Deploy it live through a **browser-based WebGPU app** that runs 24/7, server-free.

---

###  Objectives

- Demonstrate an **end-to-end pipeline** from dataset to live demo.  
- Make the entire system run **on consumer hardware** (no paid APIs or cloud GPUs).  
- Provide a pattern for **domain-specific assistants** that organizations can replicate.  
- Eventually serve as a working showcase for **transportation analytics and education**.

---


###  Architecture Overview

- **Frontend:** React + Vite (TypeScript)  
- **Model Runtime:** [`@mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm)  
  - Executes models like *Llama-3-8B-Instruct* locally in the browser via WebGPU  
- **RAG Engine:** In-browser TF-IDF retriever (no backend)  
- **Deployment:** GitHub Pages (static, free, always-on)

