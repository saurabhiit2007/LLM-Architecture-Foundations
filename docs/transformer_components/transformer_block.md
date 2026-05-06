# The Transformer Block

## 1. Overview

A transformer model is a stack of $L$ identical **transformer blocks** (also called transformer layers). Each block applies two sublayers in sequence, with a residual connection around each:

1. **Multi-Head Self-Attention** — routes information between token positions
2. **Feed-Forward Network (FFN)** — transforms each token independently

Understanding the block structure is prerequisite to understanding all other architectural variants (MoE, MQA, normalization placement, etc.).

---

## 2. The Pre-LN Block (Modern Standard)

Modern LLMs use **Pre-Layer Normalization** (Pre-LN), where normalization is applied *before* each sublayer:

$$x \leftarrow x + \text{Attn}(\text{LN}(x))$$
$$x \leftarrow x + \text{FFN}(\text{LN}(x))$$

In code:
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

The original 2017 Transformer used **Post-LN** (normalization after the residual), which is harder to train deep. See [Normalization Techniques](normalization_techniques.md) for the full comparison.

---

## 3. Residual Connections

The `x + sublayer(x)` pattern is a **residual (skip) connection** from ResNet (He et al., 2016). Their role in transformers:

**Gradient flow.** Without residuals, gradients must pass through every sublayer during backpropagation. In deep networks, this causes vanishing gradients. Residuals create a "highway" that lets gradients flow directly from the loss to early layers.

**Identity initialization.** At initialization, the sublayers (attention and FFN) are near-zero. The residual means the block starts as a near-identity function. Training then nudges the sublayers to learn *corrections* to the identity, rather than learning the full transformation from scratch. This makes optimization stable.

**Ensemble interpretation.** A stack of $L$ residual layers can be viewed as an ensemble of $2^L$ paths of different depths (some paths skip many layers). Shallower paths dominate early in training; deeper paths contribute more as training progresses.

---

## 4. Full Model Structure

```
Input token IDs  →  Token Embedding  →  [+ Positional Encoding]
                                              ↓
                              ┌─────────────────────────────┐
                              │  Transformer Block × L      │
                              │                             │
                              │  x = x + Attn(LN(x))       │
                              │  x = x + FFN(LN(x))        │
                              └─────────────────────────────┘
                                              ↓
                              Final LayerNorm (Pre-LN models)
                                              ↓
                              LM Head (Linear → Softmax → logits)
```

The LM head is a linear projection from $d_\text{model}$ to vocabulary size $V$. In most models it shares weights with the token embedding matrix (**weight tying**), halving the parameter cost of the embedding/unembedding tables.

---

## 5. Parameter Count Per Block

For a model with $d_\text{model}$, $h$ attention heads, and $d_{ff}$ FFN width:

| Component | Parameters |
|-----------|-----------|
| $Q$ projection | $d_\text{model} \times d_\text{model}$ |
| $K$ projection | $d_\text{model} \times d_\text{model}$ |
| $V$ projection | $d_\text{model} \times d_\text{model}$ |
| Output ($O$) projection | $d_\text{model} \times d_\text{model}$ |
| FFN up-projection | $d_\text{model} \times d_{ff}$ |
| FFN down-projection | $d_{ff} \times d_\text{model}$ |
| 2× RMSNorm | $2 \times d_\text{model}$ (negligible) |
| **Total per block** | $\approx 4 d_\text{model}^2 + 2 d_\text{model} d_{ff}$ |

For a standard $d_{ff} = 4 d_\text{model}$, each block has $\approx 12 d_\text{model}^2$ parameters. Total model parameters ≈ $12 L d_\text{model}^2$ (ignoring embeddings).

**Example — LLaMA 2 7B:**
$L = 32$, $d_\text{model} = 4096$, $d_{ff} = 11008$ (SwiGLU-adjusted)
$\approx 32 \times (4 \times 4096^2 + 3 \times 4096 \times 11008) \approx 6.7\text{B parameters}$

---

## 6. Depth vs Width

Given a fixed parameter budget, how should you split it between layers $L$ and hidden size $d_\text{model}$?

- **Wider (larger $d_\text{model}$):** More capacity per layer; better at capturing complex token representations
- **Deeper (more $L$):** More sequential reasoning steps; better at multi-step tasks

Empirically, scaling both together (as Chinchilla and Kaplan scaling laws describe) is optimal. Modern LLMs are typically more wide than deep relative to early models.

---

## 7. Relationship to Attention Variants

All attention variants (MQA, GQA, Sliding Window) are drop-in replacements for the attention sublayer within the same block structure. The residual and FFN remain unchanged.

| Variant | What Changes | What Stays the Same |
|---------|-------------|---------------------|
| [MQA](../attention_mechanisms/mqa.md) | Fewer KV heads | Block structure, FFN |
| [GQA](../attention_mechanisms/gqa.md) | Grouped KV heads | Block structure, FFN |
| [Sliding Window](../attention_mechanisms/sliding_window.md) | Attention span limited | Block structure, FFN |
| MoE | FFN → router + multiple FFN experts | Block structure, attention |

---

## 8. KV Cache and the Block

The KV cache stores the Key and Value tensors from every past token in every transformer block. For a model with $L$ blocks, $h_{kv}$ KV heads, and $d_h$ head dimension:

$$\text{KV cache size} = 2 \times L \times h_{kv} \times d_h \times T \times \text{bytes}$$

where $T$ is the current sequence length. This grows linearly with sequence length and is the primary memory bottleneck during long-context inference. See [LLM-Inference-Speed: KV Caching](https://saurabhiit2007.github.io/LLM-Inference-Speed/attention_optimization/kv_caching/) for the full breakdown.

---

*Sources: Vaswani et al. (2017) — Attention Is All You Need [[arXiv:1706.03762]](https://arxiv.org/abs/1706.03762) · He et al. (2016) — Deep Residual Learning [[arXiv:1512.03385]](https://arxiv.org/abs/1512.03385) · Xiong et al. (2020) — On Layer Normalization in the Transformer [[arXiv:2002.04745]](https://arxiv.org/abs/2002.04745)*
