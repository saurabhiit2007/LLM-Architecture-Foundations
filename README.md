# LLM-Architecture-Foundations

Interview prep reference for transformer architecture. Covers the building blocks of modern LLMs: how the model is structured, how attention works, how tokens are processed, and how real-world model families differ from each other.

**Live docs:** https://saurabhiit2007.github.io/LLM-Architecture-Foundations/

---

## Coverage

### Transformer Components
- Transformer Block — residual connections, Pre-LN structure, parameter count
- Normalization — LayerNorm, RMSNorm, Pre-LN vs Post-LN
- Activation Functions — GELU, SwiGLU, GeGLU, GLU family
- Feed-Forward Network (FFN) — 4× expansion, GLU adjustment, associative memory view

### Attention Mechanisms
- Multi-Head Attention (MHA) — Q/K/V projections, scaled dot-product, O(n²) complexity
- Multi-Query Attention (MQA) — single KV head, inference memory savings
- Grouped-Query Attention (GQA) — spectrum between MHA and MQA, used in LLaMA 3
- Sliding Window Attention — O(n·W) complexity, StreamingLLM, attention sinks
- **KV Cache** → see [LLM-Inference-Speed](https://saurabhiit2007.github.io/LLM-Inference-Speed/attention_optimization/kv_caching/)

### Positional Encodings
- Sinusoidal (Original) — sin/cos formula, relative position via dot products, limitations
- RoPE — rotation in complex space, context extension (YaRN), used in LLaMA/Mistral
- ALiBi — linear bias on attention logits, length extrapolation, used in BLOOM/MPT

### Tokenization
- Byte Pair Encoding (BPE) — merge rules, frequency-based, used in GPT/LLaMA 3
- Unigram — probabilistic, EM pruning, comparison to BPE
- SentencePiece Unigram — framework vs algorithm, subword regularization
- TikToken — OpenAI BPE implementation, cl100k_base, o200k_base
- Vocabulary Size Tradeoffs — sequence length impact, embedding table cost, fertility, multilingual

### Model Architecture Families
- Encoder / Decoder / Encoder-Decoder — attention type, objective, use case, KV cache usage
- Dense vs Sparse Models — parameter efficiency, routing, activation sparsity
- Mixture of Experts (MoE) — top-k routing, expert capacity, load balancing, training dynamics

### Pretraining Objective
- Next Token Prediction — CLM objective, causal mask, teacher forcing, perplexity, comparison to MLM/span corruption

### Scaling Laws
- Chinchilla & Compute-Optimal Training — Kaplan power laws, 20 tokens/param rule, over-training rationale, emergent capabilities

### Real-World Model Families
- Claude — Haiku/Sonnet/Opus tiers, Constitutional AI, System 1/2 thinking
- LLaMA — Meta recipe (over-training, distillation, DPO), GQA, RoPE scaling, 128K tokenizer
- Gemini — native multimodal training, dynamic reasoning, RLAIF
- OpenAI — GPT-2/3 architecture, GPT-4 MoE, o1/o3 RLVR reasoning
- Mistral — 7B architecture (GQA, SWA, RoPE, RMSNorm, SwiGLU), Mixtral 8×7B MoE
- DeepSeek — MLA (latent KV compression), fine-grained MoE, MTP, R1 reasoning via RL
- Dense vs MoE Comparison — inference cost, training cost, when to use each
- General Architecture Comparison — decision matrix, the 2024-2025 standard stack

---

## Related Repos

| Repo | Focus |
|------|-------|
| [LLM-Inference-Speed](https://github.com/saurabhiit2007/LLM-Inference-Speed) | KV cache, FlashAttention, quantization, serving frameworks |
| [LLM-Training-Hub](https://github.com/saurabhiit2007/LLM-Training-Hub) | Pre/mid/post-training, LoRA, distributed training |
| [LLM-Alignment-Reasoning](https://github.com/saurabhiit2007/LLM-Alignment-Reasoning) | RLHF, DPO, Constitutional AI, CoT, reasoning models |
| [RAG-and-Agents](https://github.com/saurabhiit2007/RAG-and-Agents) | RAG pipeline, agent frameworks, context engineering |

---

## Local Development

```bash
# Install dependencies
uv sync

# Serve with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```
