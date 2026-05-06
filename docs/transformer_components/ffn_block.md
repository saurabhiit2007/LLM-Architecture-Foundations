# Feed-Forward Network (FFN) Block

## 1. Overview

The Feed-Forward Network (FFN) is one of the three core components of every transformer layer, alongside multi-head attention and normalization. It operates **independently on each token** (no cross-token interaction) and accounts for roughly two-thirds of a transformer's parameters.

The two sublayers of a transformer block:
1. **Multi-Head Attention** — token mixing (routes information between positions)
2. **FFN** — per-token transformation (stores and retrieves factual knowledge)

---

## 2. Structure

### Standard FFN

$$\text{FFN}(x) = W_2 \cdot \text{act}(W_1 x + b_1) + b_2$$

Two linear projections with a nonlinear activation in between:

1. **Expand**: project from $d_\text{model}$ → $d_{ff}$ (up-projection)
2. **Activate**: apply nonlinearity (GELU, SwiGLU, etc.)
3. **Contract**: project from $d_{ff}$ → $d_\text{model}$ (down-projection)

### GLU Variant FFN (SwiGLU / GeGLU)

$$\text{FFN}_\text{SwiGLU}(x) = \bigl(\text{Swish}(xW_\text{gate}) \odot xW_\text{up}\bigr) W_\text{down}$$

Three matrices instead of two: gate projection, up projection, and down projection. The gate controls which information flows through.

---

## 3. The 4× Expansion Ratio

By convention, $d_{ff} = 4 \times d_\text{model}$.

| Model | $d_\text{model}$ | $d_{ff}$ | Ratio |
|-------|-----------------|---------|-------|
| GPT-3 175B | 12,288 | 49,152 | 4× |
| LLaMA 2 7B | 4,096 | 11,008 | ≈2.7× (SwiGLU adjusted) |
| LLaMA 3 8B | 4,096 | 14,336 | 3.5× (SwiGLU adjusted) |

The 4× ratio is empirical — it provides enough capacity for the FFN to function as an associative memory without excessive compute. There is no mathematical derivation; it was set in the original Transformer paper and has been widely replicated.

---

## 4. GLU Adjustment: Why (2/3) × 4×

GLU variants require **three weight matrices** instead of two. To keep the parameter count equal to a standard 4× FFN, the expansion dimension is reduced:

$$d_{ff}^{\text{GLU}} = \frac{2}{3} \times 4 \times d_\text{model} = \frac{8}{3} d_\text{model}$$

**Why (2/3)?** A standard FFN uses $2 \times d_\text{model} \times d_{ff}$ parameters. A GLU FFN uses $3 \times d_\text{model} \times d_{ff}^{\text{GLU}}$ parameters. Setting these equal: $d_{ff}^{\text{GLU}} = \frac{2}{3} d_{ff}$.

In practice, implementations round to a multiple of 64 or 256 for hardware efficiency.

---

## 5. FFN as Associative Memory

Geva et al. (2021) showed that FFN layers function as **key-value memory stores**:
- The rows of $W_1$ are **keys** — they pattern-match input features
- The rows of $W_2$ are **values** — they store what to output when a key fires
- The activation function determines which memories are "active"

This explains why knowledge editing (changing what the model knows about a specific fact) often targets FFN weights rather than attention weights.

---

## 6. Parameter Count

For a single transformer layer with $d_\text{model}$ and $d_{ff}$:

| Component | Parameters |
|-----------|-----------|
| Attention ($Q, K, V, O$ projections) | $4 \times d_\text{model}^2$ |
| FFN (standard, 2 matrices) | $2 \times d_\text{model} \times d_{ff}$ |
| FFN (GLU variant, 3 matrices) | $3 \times d_\text{model} \times d_{ff}^{\text{GLU}}$ |
| LayerNorm / RMSNorm (2 per layer) | $4 \times d_\text{model}$ (negligible) |

For a standard FFN with $d_{ff} = 4 d_\text{model}$, the FFN has **twice** the parameters of the attention block.

---

## 7. Comparison: Attention vs FFN

| Aspect | Multi-Head Attention | FFN |
|--------|---------------------|-----|
| Cross-token interaction | Yes (all positions attend to each other) | No (each token processed independently) |
| Role | Token routing / mixing | Per-token knowledge retrieval |
| Complexity | $O(n^2 \cdot d)$ | $O(n \cdot d \cdot d_{ff})$ |
| Dominant in long contexts | Yes | No |
| Knowledge storage | Minimal | Primary |

---

*Sources: Vaswani et al. (2017) — Attention Is All You Need [[arXiv:1706.03762]](https://arxiv.org/abs/1706.03762) · Geva et al. (2021) — Transformer Feed-Forward Layers Are Key-Value Memories [[arXiv:2012.14913]](https://arxiv.org/abs/2012.14913) · Shazeer (2020) — GLU Variants Improve Transformer [[arXiv:2002.05202]](https://arxiv.org/abs/2002.05202)*
