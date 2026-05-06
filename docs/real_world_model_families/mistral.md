# Mistral Model Family

## 1. Model Overview

Mistral AI has released two architecturally distinct model lines: the **Mistral** dense models and the **Mixtral** sparse MoE models. Both are open-weight and have been influential in demonstrating high capability at smaller parameter counts.

| Model | Params (active) | Architecture | Key Features |
|-------|----------------|-------------|--------------|
| Mistral 7B | 7B | Dense decoder-only | SWA, GQA, no absolute PE |
| Mixtral 8x7B | 13B active / 47B total | Sparse MoE | 8 experts, top-2 routing |
| Mixtral 8x22B | 39B active / 141B total | Sparse MoE | Larger MoE, multilingual |
| Mistral Large | Not disclosed | Dense | Flagship closed model |

---

## 2. Mistral 7B Architecture

Mistral 7B (2023) introduced two attention optimizations as defaults that have since become industry-standard:

**Grouped Query Attention (GQA):**

- Uses 8 KV heads instead of 32 (same as query heads)
- Reduces KV cache memory by 4× at inference
- See [GQA](../attention_mechanisms/gqa.md) for full details

**Sliding Window Attention (SWA):**

- Each token attends only to the previous 4,096 tokens (window size $W = 4096$)
- Reduces attention complexity from $O(n^2)$ to $O(n \cdot W)$
- Multi-layer stacking gives a theoretical receptive field of $W \times \text{depth}$ tokens
- See [Sliding Window Attention](../attention_mechanisms/sliding_window.md)

**Other architectural choices:**

- RoPE positional encoding (no learned absolute embeddings)
- RMSNorm (not LayerNorm)
- SwiGLU activation in the FFN
- Vocabulary: 32,000 (SentencePiece, same tokenizer as LLaMA 2)

**Mistral 7B config:**

| Component | Value |
|-----------|-------|
| Layers | 32 |
| Hidden dim | 4,096 |
| Query heads | 32 |
| KV heads (GQA) | 8 |
| FFN dim | 14,336 |
| Context length | 8,192 (SWA window: 4,096) |

---

## 3. Mixtral 8x7B: Sparse MoE

Mixtral 8x7B replaces each dense FFN layer with a **Mixture of Experts** block containing 8 expert FFNs. A router selects **top-2 experts** per token.

**Why this matters:**

- **Active parameters**: Only ~13B parameters are used per forward pass (2 of 8 experts per layer)
- **Total parameters**: ~47B stored (all 8 experts)
- Inference cost matches a ~13B dense model; quality approaches a ~70B dense model

**Architecture:**

- Attention layers are **identical to Mistral 7B** (GQA, SWA, RoPE)
- Only the FFN layers become MoE blocks
- Router: linear layer producing logits over 8 experts; softmax-weighted sum of top-2

**Context length:** 32K tokens (SWA window extended vs 7B)

See [Mixture of Experts](../model_architecture_families/mixture_of_experts.md) for routing and load balancing details.

---

## 4. Performance vs Size

Mistral 7B was notable at release for outperforming LLaMA 2 13B on most benchmarks while being nearly half the size. The key driver was the architectural efficiency (GQA + SWA) rather than more data or larger scale.

Mixtral 8x7B matches or exceeds LLaMA 2 70B on most benchmarks at a fraction of the inference cost, demonstrating that sparse MoE is an efficient path to high capability.

---

## 5. Key Architectural Contributions

- **GQA as default** — Mistral 7B normalized GQA as the expected choice for efficient inference; adopted subsequently by LLaMA 3, Gemma, and others
- **SWA as production feature** — Demonstrated SWA works well in practice for long-context tasks
- **Open-weight MoE** — Mixtral made sparse MoE accessible for fine-tuning and research; previously only GPT-4 (suspected) and Google's models used MoE at this scale

---

*Sources: Jiang et al. (2023) — Mistral 7B [[arXiv:2310.06825]](https://arxiv.org/abs/2310.06825) · Jiang et al. (2024) — Mixtral of Experts [[arXiv:2401.04088]](https://arxiv.org/abs/2401.04088)*
