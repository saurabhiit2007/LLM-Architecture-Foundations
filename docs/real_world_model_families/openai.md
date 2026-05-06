# OpenAI Model Family

## 1. Model Overview

OpenAI has released several distinct model lines since GPT-2. Key architectural generations and their notable design decisions:

| Model | Year | Parameters | Key Architecture Choices |
|-------|------|-----------|--------------------------|
| GPT-2 | 2019 | 1.5B | Decoder-only, learned absolute PE, Pre-LN |
| GPT-3 | 2020 | 175B | Same architecture, massively scaled |
| InstructGPT | 2022 | 175B | GPT-3 + RLHF (SFT → RM → PPO) |
| GPT-4 | 2023 | ~1.8T est. | Multimodal, likely MoE, not disclosed |
| o1 / o1-mini | 2024 | Not disclosed | Extended thinking, RLVR training |
| o3 / o3-mini | 2025 | Not disclosed | Improved reasoning over o1 |

---

## 2. GPT-2 and GPT-3 Architecture

Both are standard **decoder-only Transformers**. Key choices that became widely influential:

- **Pre-Layer Normalization (Pre-LN)**: LayerNorm applied *before* attention and FFN sublayers rather than after. This stabilizes gradient flow at scale and is now the default in nearly all modern LLMs.
- **Learned absolute positional embeddings**: Position vectors trained as parameters. Later superseded by RoPE in most architectures.
- **No architectural change from GPT-2 → GPT-3**: GPT-3 proved that scaling the same architecture with more data delivers dramatic capability gains without architectural innovation.

**GPT-3 at scale (175B):**

| Component | Value |
|-----------|-------|
| Layers | 96 |
| Hidden dim | 12,288 |
| Attention heads | 96 |
| Context length | 2,048 |
| Vocabulary | 50,257 (BPE) |

---

## 3. GPT-4

OpenAI has not published GPT-4's architecture. Based on published analysis:

- Believed to be a **Mixture of Experts (MoE)** model with ~16 experts per layer
- **Multimodal** from launch (vision encoder + language decoder)
- Context window extended to 128K tokens via RoPE scaling
- Trained with RLHF and rule-based reward signals

**GPT-4 Technical Report** (2023) deliberately withholds architectural details but reports extensive benchmark results.

---

## 4. o1 / o3: Reasoning Models

The o-series represents a paradigm shift: rather than scaling model size, OpenAI trained models to generate **extended reasoning chains at inference time** before producing an answer.

**Key differences from GPT-4:**

- Extended chain-of-thought is generated internally (hidden from user by default in the API)
- Training uses **Reinforcement Learning with Verifiable Rewards (RLVR)** — rewarded for correct final answers on math/code, not for reasoning style
- More inference compute → better answers (test-time compute scaling)
- Trades **latency** for **accuracy** on hard reasoning tasks

**Benchmark results:**

| Benchmark | GPT-4o | o1 |
|-----------|--------|-----|
| AIME 2024 (math olympiad) | 13% | 83% |
| GPQA Diamond (PhD science) | 56% | 78% |
| Codeforces Elo | ~800 | ~1,673 |

---

## 5. TikToken and Vocabulary

OpenAI uses **TikToken** (BPE-based) across all models. Vocabulary has grown to improve efficiency for code and non-English text:

| Encoding | Models | Vocab size |
|----------|--------|-----------|
| `p50k_base` | GPT-3, Codex | 50,257 |
| `cl100k_base` | GPT-3.5, GPT-4 | 100,277 |
| `o200k_base` | GPT-4o, o1, o3 | 200,019 |

See [TikToken](../tokenization/tiktoken.md) for tokenization details.

---

## 6. Key Architectural Contributions

- **Pre-LN** — Stabilizes training at scale; adopted by virtually all subsequent LLMs
- **Scale as a capability lever** — GPT-3 showed architecture need not change to achieve step-change capability
- **RLHF pipeline** — InstructGPT established SFT → RM → PPO, now the standard post-training recipe
- **Test-time compute scaling** — o1/o3 showed that inference-time reasoning chains can substitute for larger base models on hard tasks

---

*Sources: Brown et al. (2020) — GPT-3 [[arXiv:2005.14165]](https://arxiv.org/abs/2005.14165) · OpenAI (2023) — GPT-4 Technical Report [[arXiv:2303.08774]](https://arxiv.org/abs/2303.08774) · OpenAI (2024) — o1 System Card [[openai.com]](https://openai.com/index/openai-o1-system-card/)*
