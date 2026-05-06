# ALiBi (Attention with Linear Biases)

## 1. Overview

ALiBi (Attention with Linear Biases) is a positional encoding method introduced by Press et al. (2022) that replaces learned or rotary position embeddings with a simple **linear penalty added directly to attention logits**. Its key property is strong **length extrapolation**: a model trained on sequences of length 1024 can generalize to sequences of length 2048 or longer at inference time without any fine-tuning.

Used in: **BLOOM** (176B), **MPT**, and several other open models.

---

## 2. Core Idea

Standard attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

ALiBi adds a fixed, non-learned bias matrix to the pre-softmax scores:

$$\text{Attention} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot B\right)V$$

Where:

- $B_{ij} = -(i - j)$ for $j \leq i$ — the distance between query position $i$ and key position $j$
- $m$ is a **head-specific scalar slope** (fixed, not learned)

The bias penalizes attending to distant tokens proportionally to their distance. The further away a token is, the more its logit is suppressed.

---

## 3. Head-Specific Slopes

Each attention head gets a different slope $m$, forming a geometric sequence. For $n$ heads:

$$m_h = 2^{-8h/n}, \quad h = 1, \ldots, n$$

For 8 heads, slopes are $\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \ldots, \frac{1}{256}$

- **Steep slope (large $m$)**: head strongly prefers nearby tokens — local attention behavior
- **Shallow slope (small $m$)**: head can attend more globally — long-range dependencies

This gives the model a spectrum of attention ranges across heads with zero learned parameters.

---

## 4. Why It Extrapolates

Standard positional encodings (absolute or RoPE) struggle beyond their training context length because the model has never seen position indices larger than its training maximum. ALiBi avoids this by using **distance** rather than absolute position — a query-key pair 500 tokens apart receives the same bias whether it sits at positions (100, 600) or (1000, 1500). The bias function generalizes naturally to any distance.

**Key result (Press et al.):** A model trained with ALiBi at 1024 tokens matches or exceeds a RoPE baseline trained at 2048 tokens when evaluated at 2048 — it extrapolates past its training length better than RoPE trained directly at the target length.

---

## 5. ALiBi vs RoPE

| Dimension | ALiBi | RoPE |
|-----------|-------|------|
| Mechanism | Additive bias on attention logits | Rotates Q and K vectors |
| Learned parameters | None (slopes are fixed) | None (frequencies are fixed) |
| Length extrapolation | Strong out-of-the-box | Requires YaRN / RoPE scaling tricks |
| Relative position | Yes (distance-based) | Yes (via rotation composition) |
| Used in | BLOOM, MPT | LLaMA, Mistral, PaLM, Gemini |
| Decay behavior | Linear with distance | Complex sinusoidal |

**Prefer ALiBi** when length generalization without fine-tuning is the priority.

**Prefer RoPE** when raw benchmark performance matters — RoPE dominates frontier models (2024–2025), and YaRN/LongRoPE have largely closed the extrapolation gap.

---

## 6. Limitations

- **Distance only, no absolute position** — ALiBi encodes how far apart two tokens are, but not where they are absolutely. Tasks requiring absolute position awareness may suffer.
- **Not dominant in frontier models** — LLaMA, Mistral, Gemini, and GPT-4 all use RoPE variants. RoPE with context extension techniques has largely matched ALiBi's extrapolation advantage.
- **Causal masking interaction** — The bias applies after the causal mask; requires careful implementation to avoid off-by-one errors at sequence boundaries.

---

*Source: Press et al. (2022) — Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation [[arXiv:2108.12409]](https://arxiv.org/abs/2108.12409)*
