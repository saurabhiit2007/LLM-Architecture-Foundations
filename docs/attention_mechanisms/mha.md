# Multi-Head Attention (MHA)

## 1. Overview

Multi-Head Attention is the fundamental mechanism that allows Transformers to focus on different parts of an input sequence simultaneously. Instead of one "viewpoint," the model gets multiple parallel perspectives.

---

## 2. The Mathematical Mechanism

The MHA process involves transforming input embeddings into three distinct spaces: **Queries (Q)**, **Keys (K)**, and **Values (V)**.

### The Step-by-Step Flow

1. **Linear Projection**: For $h$ heads, the input $X$ is projected using learned weights $W_i^Q, W_i^K, W_i^V$.
2. **Scaled Dot-Product Attention**: Each head computes attention independently:
   $$\text{Head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

3. **Concatenation**: All heads are joined: $\text{Concat}(\text{Head}_1, ..., \text{Head}_h)$.
4. **Final Output Projection**: The result is multiplied by $W^O$ to return to the original model dimension.

---

## 3. Why $\sqrt{d_k}$ Scaling Is Necessary

Without scaling, dot products grow with dimensionality. If each component of $Q$ and $K$ has variance 1, then $\text{Var}(Q \cdot K) = d_k$, meaning typical magnitudes scale as $\sqrt{d_k}$. Large logits push softmax toward one-hot distributions, collapsing gradient flow. Dividing by $\sqrt{d_k}$ normalizes variance back to 1 regardless of head dimension.

---

## 4. Complexity and Parallelism

- **Time and memory**: $O(n^2 \cdot d)$ per layer — the quadratic bottleneck that motivates [Sliding Window Attention](sliding_window.md) and FlashAttention
- **Training parallelism**: All heads and all sequence positions computed simultaneously (unlike RNNs)
- **Inference**: Autoregressive decoding is sequential per token, but each step uses the KV cache to avoid recomputing past tokens

---

## 5. Key Extensions

| Extension | What it changes | Where to read |
|-----------|----------------|---------------|
| MQA / GQA | Fewer KV heads → smaller KV cache | [MQA](mqa.md), [GQA](gqa.md) |
| Sliding Window | Limits attention span → $O(n \cdot W)$ | [Sliding Window](sliding_window.md) |
| RoPE | Rotates Q/K to encode relative position | [RoPE](../positional_encodings/RoPE.md) |
| KV Cache | Stores past K/V to avoid recomputation at inference | [LLM-Inference-Speed: KV Caching](https://saurabhiit2007.github.io/LLM-Inference-Speed/attention_optimization/kv_caching/) |
| FlashAttention | Tiled SRAM kernels → faster, memory-efficient | [LLM-Inference-Speed: FlashAttention](https://saurabhiit2007.github.io/LLM-Inference-Speed/attention_optimization/flash_attention/) |

---
