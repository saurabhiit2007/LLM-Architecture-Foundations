# Sinusoidal Positional Encoding

## 1. Overview

Sinusoidal positional encoding is the original method introduced in "Attention is All You Need" (Vaswani et al., 2017). It adds a fixed, non-learned vector to each token embedding to encode the token's position in the sequence. It is the baseline that motivated all subsequent approaches ([RoPE](RoPE.md), [ALiBi](ALiBi.md)).

The problem it solves: self-attention is **permutation-invariant** — without positional information, "The cat sat on the mat" and "The mat sat on the cat" produce the same attention outputs.

---

## 2. The Formula

For position $pos$ and embedding dimension index $i$:

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

- Even dimensions use $\sin$, odd dimensions use $\cos$
- The wavelength $\lambda_i = 2\pi \cdot 10000^{2i/d_\text{model}}$ grows geometrically with $i$
- Dimension 0: wavelength ≈ 6.3 (changes rapidly with position)
- Dimension $d_\text{model}/2$: wavelength ≈ 62,832 (changes slowly)

The resulting vector $\text{PE}(pos) \in \mathbb{R}^{d_\text{model}}$ is added to the token embedding before the first transformer layer.

---

## 3. Why Sine and Cosine?

**Relative position as a linear function.** Using trigonometric identities:

$$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$
$$\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)$$

This means $\text{PE}(pos + k)$ can be expressed as a **linear transformation** of $\text{PE}(pos)$. As a result, the dot product $\text{PE}(pos) \cdot \text{PE}(pos + k)$ depends only on the offset $k$, not on the absolute position. The model can in principle learn to use relative distances from these patterns.

**Bounded values.** All components are in $[-1, 1]$, so adding the encoding doesn't change the scale of token embeddings.

**No training required.** The encoding is deterministic — any sequence length can be encoded at inference time without retraining.

---

## 4. Geometric Intuition

Think of each pair of dimensions $(2i, 2i+1)$ as a clock hand rotating at a different frequency:
- Low dimensions: fast clock — distinguishes nearby positions
- High dimensions: slow clock — distinguishes far-apart positions

Together, the full vector forms a unique "fingerprint" for each position, analogous to binary counting where each bit represents a different scale.

---

## 5. Limitations

**Poor length extrapolation.** The encoding is defined for any position, but the model only sees positions up to the training context length. At inference time, tokens at positions beyond the training maximum receive encodings the model was never trained on, causing degraded performance.

**Absolute, not relative.** The encoding encodes the absolute position $pos$, not the relative distance between two tokens. The relative distance can be recovered via dot products, but this puts the burden on the model to learn this relationship rather than encoding it directly. [RoPE](RoPE.md) and [ALiBi](ALiBi.md) encode relative position explicitly.

**No learned adaptation.** A fixed encoding cannot adapt to the statistics of a specific domain or language.

---

## 6. Learned Positional Embeddings

The alternative to sinusoidal PE is to learn a position embedding table $E \in \mathbb{R}^{L_\text{max} \times d_\text{model}}$ during training. Used in GPT-2, BERT.

| Dimension | Sinusoidal | Learned |
|-----------|-----------|---------|
| Learned parameters | None | $L_\text{max} \times d_\text{model}$ |
| Extrapolation | Defined but unreliable | Not possible (no embedding for unseen positions) |
| Short sequences | Good | Good |
| Long sequences | Degrades | Hard cap at $L_\text{max}$ |
| Current usage | Rare | Rare (replaced by RoPE) |

Both approaches have been largely superseded by [RoPE](RoPE.md) in modern LLMs because RoPE encodes relative position directly in the attention computation and scales better to long contexts.

---

## 7. Comparison Summary

| Encoding | Relative Position | Extrapolation | Parameters | Used In |
|----------|-----------------|---------------|-----------|---------|
| Sinusoidal | Via dot products | Defined, unreliable | None | Original Transformer |
| Learned | No | None beyond $L_\text{max}$ | $L \times d$ | GPT-2, BERT |
| [RoPE](RoPE.md) | Explicit (rotation) | Good with scaling tricks | None | LLaMA, Mistral, GPT-4 |
| [ALiBi](ALiBi.md) | Explicit (linear bias) | Strong out-of-box | None | BLOOM, MPT |

---

*Source: Vaswani et al. (2017) — Attention Is All You Need [[arXiv:1706.03762]](https://arxiv.org/abs/1706.03762)*
