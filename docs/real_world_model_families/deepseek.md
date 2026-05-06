# DeepSeek Model Family

## 1. Overview

DeepSeek (by DeepSeek AI, China) became one of the most significant model families of 2024–2025 by achieving near-frontier performance at a fraction of the training cost of comparable Western models. Two key architectural innovations drove this: **MLA** (Multi-head Latent Attention) and **MTP** (Multi-Token Prediction).

| Model | Released | Parameters (Total / Active) | Key Innovation |
|-------|----------|----------------------------|---------------|
| DeepSeek-V2 | May 2024 | 236B / 21B | MLA, MoE |
| DeepSeek-V3 | Dec 2024 | 671B / 37B | MTP, FP8 training |
| DeepSeek-R1 | Jan 2025 | 671B / 37B | GRPO reasoning, no supervised CoT |

---

## 2. MLA: Multi-head Latent Attention

MLA is DeepSeek's replacement for standard GQA/MQA. Instead of reducing the *number* of KV heads, MLA compresses the *dimension* of each K and V via a learned low-rank projection.

### How It Works

**Standard MHA:** KV cache stores $h$ full-dimension K and V vectors per token.

**MLA:** The model first compresses the hidden state into a low-dimensional **latent vector**, then expands it back to full K and V at attention time:

$$c_t^{KV} = W^{DKV} h_t \quad \in \mathbb{R}^{d_c}$$

$$K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV}$$

Where:
- $d_c \ll d_\text{model}$ (latent dimension, much smaller than full hidden dim)
- $W^{DKV}$ is the down-projection (compression)
- $W^{UK}, W^{UV}$ are up-projections (expand to K and V at query time)

**What gets cached:** Only the compressed latent $c_t^{KV}$ (not the full K/V), reducing KV cache memory significantly compared to even GQA.

| Method | KV Cache Size per Token |
|--------|------------------------|
| MHA (h=128) | $2 \times 128 \times d_h$ |
| GQA (g=8) | $2 \times 8 \times d_h$ |
| MLA (DeepSeek-V2) | $1 \times d_c$ where $d_c \approx 4 \times d_h$ |

For DeepSeek-V2, MLA reduces KV cache by ~5.75× compared to MHA while matching or exceeding MHA quality.

---

## 3. DeepSeek MoE Architecture

Both V2 and V3 use a fine-grained MoE where each FFN layer has many small experts:

- **V2**: 160 experts per layer, top-2 routing, 2 shared experts always active
- **V3**: 256 experts, top-8 routing, 1 shared expert

**Shared experts** are a DeepSeek innovation: a small number of experts are always active for every token. This ensures a stable "backbone" of common knowledge while specialized experts handle domain-specific patterns.

DeepSeek V3 also introduced an **auxiliary-loss-free** load balancing strategy — instead of adding a separate load balancing loss term (which can interfere with training), they use a bias term on expert routing scores to nudge load distribution.

---

## 4. MTP: Multi-Token Prediction

DeepSeek-V3 introduced MTP as an auxiliary training objective. Instead of predicting only the next token, the model predicts the **next $k$ tokens** simultaneously.

$$\mathcal{L}_\text{MTP} = \sum_{t=1}^{T} \sum_{j=1}^{k} \log p_\theta(x_{t+j} \mid x_{<t})$$

**Benefits:**
- Denser training signal per token (more gradient information per forward pass)
- The model must represent more future context, improving long-range coherence
- At inference time, MTP heads can be reused for **speculative decoding** (predict 2+ tokens, verify cheaply)

---

## 5. DeepSeek-R1: Reasoning via RL

DeepSeek-R1 (Jan 2025) demonstrated that **strong chain-of-thought reasoning can emerge from RL alone**, without requiring supervised CoT data.

### Training Pipeline

**Stage 1 — Cold Start:** Fine-tune on a small set of long-form reasoning examples with explicit `<think>` tags to initialize the model's reasoning format.

**Stage 2 — GRPO Reasoning RL:** Apply GRPO (Group Relative Policy Optimization — see LLM-Alignment-Reasoning repo) with a verifiable reward:
- +1 for a correct final answer (math, code judged by execution)
- 0 or negative for incorrect answers or formatting violations

No process reward model (PRM) is needed — only outcome verification.

**Stage 3 — Rejection Sampling + SFT:** Generate many reasoning traces, keep only correct ones, fine-tune again to stabilize the format.

**Stage 4 — Final RLHF:** Align the reasoning model for safety and helpfulness.

### Key Results

| Benchmark | DeepSeek-R1 | OpenAI o1 |
|-----------|------------|----------|
| AIME 2024 | 79.8% | 79.2% |
| MATH-500 | 97.3% | 96.4% |
| Codeforces (rating) | 2029 | 1891 |

DeepSeek-R1 matched o1 on most reasoning benchmarks while being fully open-weights.

**The key insight:** Language models can "discover" effective reasoning strategies through RL on outcome rewards — extended thinking chains emerge as instrumental behavior, not because they were explicitly taught.

---

## 6. Why DeepSeek Matters for Architecture Interviews

| Topic | DeepSeek Contribution |
|-------|----------------------|
| Attention efficiency | MLA: KV compression via low-rank latent, better than GQA |
| MoE design | Fine-grained experts + shared experts + auxiliary-loss-free balancing |
| Training efficiency | FP8 training in V3; ~$6M total training cost vs ~$100M for GPT-4 |
| Reasoning | R1: emergent CoT from RL without SFT data |
| Open weights | Full model weights released, enabling research |

---

*Sources: DeepSeek-AI (2024) — DeepSeek-V2 [[arXiv:2405.04434]](https://arxiv.org/abs/2405.04434) · DeepSeek-AI (2024) — DeepSeek-V3 [[arXiv:2412.19437]](https://arxiv.org/abs/2412.19437) · DeepSeek-AI (2025) — DeepSeek-R1 [[arXiv:2501.12948]](https://arxiv.org/abs/2501.12948)*
