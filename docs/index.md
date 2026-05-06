# LLM Architecture Foundations

The theoretical building blocks of modern large language models. This site covers the architectural components that appear in virtually every LLM interview, from attention mechanisms and positional encodings to tokenization, normalization, and real-world model families.

---

## What's Covered

**Model Architecture Families** — Encoder-only, decoder-only, and encoder-decoder designs; dense vs. sparse models; Mixture of Experts routing and training.

**Attention Mechanisms** — Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped Query Attention (GQA), and Sliding Window Attention; the KV-cache trade-offs each design makes.

**Transformer Components** — Normalization (LayerNorm, RMSNorm, Pre-LN vs. Post-LN); activation functions (GELU, SwiGLU, GLU family).

**Positional Encodings** — RoPE (rotary embeddings, context extension via YaRN); ALiBi (linear bias, length extrapolation without retraining).

**Tokenization** — Byte Pair Encoding (BPE), Unigram language model, SentencePiece, TikToken, vocabulary size trade-offs.

**Pretraining Objective** — Next-token prediction, causal language modeling, teacher forcing, and perplexity.

**Scaling Laws** — Kaplan et al. (2020) and Chinchilla (2022): how model size, dataset size, and compute budget interact.

**Real-World Model Families** — Claude, LLaMA, Gemini, GPT/OpenAI, Mistral/Mixtral; architectural choices and how they compare.

---

## How to Use This Site

Each page is a concise reference. For interview prep, start with **Attention Mechanisms** and **Transformer Components** (the most commonly tested areas), then work through **Positional Encodings** and **Tokenization**. The **Real-World Model Families** section ties everything together with concrete examples.
