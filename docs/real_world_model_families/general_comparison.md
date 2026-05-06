# General Model Family Comparison

A cross-family comparison of the key architectural decisions in modern open and closed LLMs. For per-family deep dives, see the individual pages.

---

## 1. Architecture Decision Matrix

| Model | Attn type | PE | Norm | Activation | Vocab | Context |
|-------|-----------|-----|------|-----------|-------|---------|
| GPT-3 | MHA | Learned abs. | Pre-LN | GELU | 50K | 2K |
| GPT-4 | Unknown (MoE est.) | RoPE est. | Pre-LN | Unknown | 100K | 128K |
| LLaMA 2 | GQA | RoPE | RMSNorm | SwiGLU | 32K | 4K |
| LLaMA 3 | GQA | RoPE | RMSNorm | SwiGLU | 128K | 128K |
| Mistral 7B | GQA + SWA | RoPE | RMSNorm | SwiGLU | 32K | 8K |
| Mixtral 8x7B | GQA + SWA | RoPE | RMSNorm | SwiGLU | 32K | 32K |
| Gemini 1.5 Pro | MQA (multi-query) | RoPE | RMSNorm | GeGLU | 256K | 1M |
| Claude 3/4 | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Not disclosed | 200K |

---

## 2. The Emerging Standard Stack (2024–2025)

Nearly all new open-weight models converge on the same set of choices:

- **Decoder-only architecture** — Encoder-decoder (T5, BART) is rare for generalist LLMs
- **RoPE positional encoding** — ALiBi and learned absolute PE are largely obsolete for new work
- **RMSNorm** — Replaces LayerNorm; faster (no mean subtraction), empirically equivalent
- **SwiGLU or GeGLU activation** — Replaces GELU; slightly higher quality with same compute
- **GQA** — Replaces MHA; 4-8× KV cache reduction with minimal quality loss
- **Pre-LN** — Post-LN is unstable at scale; all modern models use Pre-LN

---

## 3. Vocabulary Size Trend

| Era | Typical vocab size | Driver |
|-----|------------------|--------|
| 2020 (GPT-3) | 50K | English-focused |
| 2023 (LLaMA 2, Mistral) | 32K | Efficient, English-focused |
| 2023–2024 (GPT-4, LLaMA 3) | 100K–128K | Code + multilingual |
| 2024 (Gemini, GPT-4o) | 200K–256K | Full multilingual coverage |

Vocabulary size has grown to reduce tokenization fragmentation for code and non-Latin scripts. See [Vocabulary Size Trade-offs](../tokenization/vocabulary_size_tradeoffs.md).

---

## 4. Context Length Evolution

| Model | Year | Context length |
|-------|------|---------------|
| GPT-3 | 2020 | 2,048 |
| LLaMA 2 | 2023 | 4,096 |
| GPT-4 | 2023 | 8K → 128K |
| Mistral 7B | 2023 | 8,192 |
| Claude 2.1 | 2023 | 200K |
| LLaMA 3 | 2024 | 128K |
| Gemini 1.5 Pro | 2024 | 1M |

Context extension is achieved via RoPE scaling (YaRN, LongRoPE) rather than architectural changes.

---

## 5. Open vs Closed Models

| Dimension | Open-weight (LLaMA, Mistral) | Closed API (GPT-4, Claude, Gemini) |
|-----------|-----------------------------|------------------------------------|
| Architecture transparency | Full (paper + weights) | Minimal (benchmark reports only) |
| Fine-tuning | Yes | No (or limited via fine-tune API) |
| Deployment | Self-hosted | API only |
| Capability ceiling | Lower (resource constraints) | Higher (scale unreported) |
| Cost at scale | Lower (own infra) | Higher (API pricing) |

---

## 6. MoE Adoption

Sparse MoE has moved from research to production:

- **GPT-4** (2023): Believed to be MoE based on leaks; not confirmed by OpenAI
- **Mixtral 8x7B** (2024): First widely-used open-weight MoE LLM
- **Gemini 1.5** (2024): Confirmed MoE architecture
- **DeepSeek-V2/V3** (2024): MoE with fine-grained routing (up to 160 experts)

The trend: MoE is becoming standard for frontier models because it achieves high quality at reduced inference cost. Dense models remain dominant for smaller, locally-deployable sizes.

---

*Sources: Touvron et al. (2023) — LLaMA 2 [[arXiv:2307.09288]](https://arxiv.org/abs/2307.09288) · Dubey et al. (2024) — LLaMA 3 [[arXiv:2407.21783]](https://arxiv.org/abs/2407.21783) · Jiang et al. (2024) — Mixtral [[arXiv:2401.04088]](https://arxiv.org/abs/2401.04088)*
