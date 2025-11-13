# Transport LLM — Edge AI Demo

### Problem Framing

Modern language models are powerful, but they are also **resource-heavy**, **cloud-dependent**, and rarely optimized for narrow technical domains.  
This project began with a question:

**Can a single developer design, adapt, and deploy a transportation-focused LLM that runs entirely on consumer hardware—no servers, no APIs, no GPU clusters and at no cost?**

Transportation was selected as the domain because it spans several deep technical areas:

- Rail operations and signaling  
- Automotive systems and diagnostics  
- Public transit engineering  
- Industry standards and terminology  

The goal was to build a practical, specialized assistant that demonstrates how domain-aware AI tools can be created without relying on commercial AI platforms.

---

### Objectives

- Prototype an **end-to-end workflow** from curated dataset → model adaptation → live demo.  
- Run the entire stack on **local hardware**, minimizing cost and external dependencies.  
- Deploy a fully browser-native AI assistant using **WebGPU**, requiring no backend server.  
- Explore pathways for building **domain-specific AI tools** that organizations could replicate.

---

### Development Path and Technical Pivot

The original plan was to train and deploy a **custom fine-tuned model** through the **MLC-LLM** toolchain (Machine Learning Compilation).  
However, through experimentation it became clear that:

- MLC’s build pipeline is complex, evolving rapidly, and can be difficult to compile on cloud notebooks.  
- The WebGPU runtime is excellent for **deployment**, but not the best path for **training** or **fine-tuning**.  
- For a browser-first demo, **RAG + prompt steering** provides strong domain performance without custom training.

Because of this, the project pivoted to a more practical and stable architecture:

- Use the well-supported **Llama-3-8B-Instruct-MLC** runtime for WebGPU.  
- Implement a **local RAG engine** to supply domain context.  
- Continue exploring full fine-tuning in a future project through **Hugging Face / PEFT / LoRA**, where training workflows are mature and better documented.

This pivot reflects real engineering decision-making: choosing the most reliable and maintainable path while still meeting functional goals.

---

### Architecture Overview

- **Frontend:** React + Vite (TypeScript)  
- **Model Runtime:** [`@mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm)  
  - Executes the model entirely in the browser using WebGPU  
- **RAG Engine:** In-browser TF-IDF retrieval (no backend services)  
- **Deployment:** GitHub Pages (static, lightweight, always online)

