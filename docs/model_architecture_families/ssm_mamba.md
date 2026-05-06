# State Space Models (SSMs) and Mamba

## 1. Overview

State Space Models (SSMs) are a class of sequence models that process sequences **recurrently** rather than with full attention. They offer $O(n)$ inference cost (vs $O(n^2)$ for transformers), making them attractive for very long sequences. Mamba (Gu & Dao, 2023) is the most prominent SSM architecture in LLMs.

---

## 2. The Core SSM Equations

An SSM maintains a hidden state $h_t$ that compresses all past context:

$$h_t = A h_{t-1} + B x_t$$
$$y_t = C h_t$$

Where $A, B, C$ are learned matrices and $x_t, y_t$ are input/output at step $t$. This is a linear recurrence — structurally similar to an RNN, but with structured $A$ matrices that enable efficient parallel training.

**Training**: SSMs can be computed as a convolution over the full sequence in parallel — $O(n \log n)$ with parallel scan.

**Inference**: Purely recurrent — constant memory per step regardless of sequence length.

---

## 3. Mamba: Selective State Spaces

Standard SSMs have fixed $A, B, C$ matrices (input-independent), limiting their ability to selectively ignore irrelevant tokens. Mamba makes $B$, $C$, and the step size $\Delta$ **input-dependent**:

$$B_t = f_B(x_t), \quad C_t = f_C(x_t), \quad \Delta_t = f_\Delta(x_t)$$

This selective mechanism lets the model decide, for each token, how much to update the hidden state — analogous to an LSTM gate but more expressive.

---

## 4. SSMs vs Transformers

| Dimension | Transformer | Mamba (SSM) |
|-----------|-------------|------------|
| Sequence complexity | $O(n^2 \cdot d)$ | $O(n \cdot d)$ |
| Inference memory | Grows with $n$ (KV cache) | Constant (fixed hidden state) |
| Random access | Yes (attention to any past token) | No (compressed into state) |
| Recall of specific facts | Strong | Weaker for needle-in-haystack |
| Long context | Expensive | Efficient |
| Current frontier models | Dominant | Niche / hybrid |

---

## 5. Hybrid Models

Pure SSM models underperform transformers on tasks requiring precise retrieval of specific past tokens (e.g., "what did the user say in message 3?"). The practical response has been **hybrid architectures** that interleave attention layers with SSM layers:

- **Jamba** (AI21 Labs) — alternating transformer and Mamba layers
- **Zamba** — similar hybrid, competitive with LLaMA on benchmarks
- **Mamba-2** — improved SSM with connections to attention theory

The current consensus: pure SSMs are not ready to replace transformers for general-purpose LLMs, but hybrids offer a promising path to efficient long-context modeling.

---

*Sources: Gu & Dao (2023) — Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv:2312.00752]](https://arxiv.org/abs/2312.00752) · Lieber et al. (2024) — Jamba: A Hybrid Transformer-Mamba Language Model [[arXiv:2403.19887]](https://arxiv.org/abs/2403.19887)*
