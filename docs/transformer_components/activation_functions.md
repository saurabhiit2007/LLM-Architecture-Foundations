## 1. Overview

GLU (Gated Linear Unit) variants are gating mechanisms used in modern neural networks, particularly in transformers. They combine linear transformations with element-wise gating to control information flow.

**Core Concept: Gating**

The fundamental idea is to use one part of the input to gate (control) another part:
```
Output = Gate(x) ⊙ Transform(x)
```

where ⊙ represents element-wise multiplication.

---

---

## 2. GLU (Gated Linear Unit)

**Formula:**
```
GLU(x) = (xW + b) ⊙ σ(xV + c)
```

**Key Points:**
- Uses sigmoid (σ) as the gate
- Splits input into two linear transformations
- One path is gated by the sigmoid of the other
- Introduced in "Language Modeling with Gated Convolutional Networks" (2017)

**Characteristics:**
- Gates values between 0 and 1
- Smooth gating mechanism
- Can suppress or allow information flow

---

---

## 3. GELU (Gaussian Error Linear Unit)

**Formula:**
```
GELU(x) = x · Φ(x)
```
where Φ(x) is the cumulative distribution function of standard normal distribution.

**Approximation (commonly used):**
```
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

<figure markdown>
  ![GELU](../images/gelu.png){ width=500 }
  <figcaption>GELU v/s ReLU</figcaption>
</figure>

**Key Points:**
- Not strictly a GLU variant, but often grouped with them
- Used in BERT, GPT-2, GPT-3
- Smooth, non-monotonic activation
- Stochastic regularizer interpretation: multiplies input by Bernoulli distributed mask

**Why it matters:**
- Default activation in many modern transformers
- Better gradient flow than ReLU in deep networks

---

## 3. SwiGLU (Swish-Gated Linear Unit)

**Formula:**
```
SwiGLU(x) = Swish(xW) ⊙ (xV)
```
where Swish(x) = x · σ(βx)

**Key Points:**
- Combines Swish activation with gating
- Used in PaLM, LLaMA models
- β is typically set to 1
- Empirically outperforms other GLU variants in large language models

**Why it's important:**
- State-of-the-art for LLMs
- Better performance-to-parameter ratio
- Industry standard in many modern architectures

---

## 4. GeGLU (GELU-Gated Linear Unit)

**Formula:**
```
GeGLU(x) = GELU(xW) ⊙ (xV)
```

**Key Points:**
- Uses GELU as the gating function
- Proposed in "GLU Variants Improve Transformer" (2020)
- Slightly better than standard FFN in transformers
- More common in vision transformers

---

## 5. ReGLU (ReLU-Gated Linear Unit)

**Formula:**
```
ReGLU(x) = ReLU(xW) ⊙ (xV)
```

**Key Points:**
- Uses ReLU as gating function
- Simpler and faster than smooth variants
- Good baseline for comparison
- Less commonly used in practice than SwiGLU/GeGLU

---

## Implementation Consideration

**Parameter Count:**
GLU variants require **~1.5x parameters** compared to standard activations because they split the input into two paths.

Example FFN comparison:
```python
# Standard FFN
hidden = activation(x @ W1)  # W1: d_model → d_ff
output = hidden @ W2          # W2: d_ff → d_model

# GLU variant FFN
gate = activation(x @ W_gate)   # W_gate: d_model → d_ff
value = x @ W_value             # W_value: d_model → d_ff
hidden = gate * value           # Element-wise multiplication
output = hidden @ W2            # W2: d_ff → d_model
```

To maintain the same parameter count, implementations often use:
- d_ff = (2/3) × original_d_ff when using GLU variants

---

## Comparison Summary

| Variant | Gate Function | Used In | Characteristics |
|---------|--------------|---------|-----------------|
| GLU | Sigmoid | CNNs, early work | Smooth gating, 0-1 range |
| GELU | - | BERT, GPT-2/3 | Smooth, probabilistic interpretation |
| SwiGLU | Swish | LLaMA, PaLM | SOTA for LLMs, best empirical performance |
| GeGLU | GELU | Vision Transformers | Good for transformers, smooth |
| ReGLU | ReLU | Baseline | Simple, sparse activation |

---

## Key Takeaways for Interviews

1. **Why GLU variants?** They provide learnable, dynamic gating that helps models control information flow more effectively than static activations.

2. **Trade-off:** Better performance but ~1.5x parameters (or reduced hidden dimension for same param count).

3. **Current best practice:** SwiGLU is the most common choice for modern LLMs (LLaMA, PaLM, etc.).

4. **When to use:**
   - SwiGLU: Language models, general-purpose transformers
   - GeGLU: Vision transformers, when you want GELU-like smoothness
   - GELU: Simpler architectures without gating overhead

5. **Mathematical intuition:** Gating allows the network to learn which activations to pass through and which to suppress, providing more expressiveness than fixed activation functions.

---

## References

- "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017)
- "GLU Variants Improve Transformer" (Shazeer, 2020)
- "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
- LLaMA paper (Touvron et al., 2023)