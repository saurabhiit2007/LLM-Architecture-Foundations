# Dense vs MoE: Real-World Comparison

This page compares dense and sparse MoE model families on practical dimensions. For the architectural concepts, see [Dense vs Sparse Models](../model_architecture_families/dense_vs_sparse_models.md) and [Mixture of Experts](../model_architecture_families/mixture_of_experts.md).

---

## 1. Representative Models

| Model | Type | Active Params | Total Params | Notes |
|-------|------|--------------|-------------|-------|
| LLaMA 3 8B | Dense | 8B | 8B | Meta, GQA, RoPE |
| LLaMA 3 70B | Dense | 70B | 70B | Meta, GQA, RoPE |
| Mistral 7B | Dense | 7B | 7B | GQA + SWA |
| Gemma 2 9B | Dense | 9B | 9B | Google, alternating attention |
| Mixtral 8x7B | Sparse MoE | ~13B | ~47B | Top-2 routing, 8 experts |
| Mixtral 8x22B | Sparse MoE | ~39B | ~141B | Top-2 routing |
| DeepSeek-V2 | Sparse MoE | ~21B | ~236B | 160 experts, top-6 |
| Grok-1 | Sparse MoE | ~87B | ~314B | 8 experts per layer |

---

## 2. Inference Cost Comparison

The key MoE advantage: inference cost scales with **active** parameters, not total parameters.

| Metric | Dense 70B | Mixtral 8x7B (~13B active) |
|--------|-----------|---------------------------|
| Memory (weights, FP16) | ~140 GB | ~94 GB (all experts loaded) |
| FLOPS per token | ~140B | ~26B (2 experts × 7B) |
| Throughput (relative) | 1× | ~3-5× faster per token |
| Quality (MMLU) | ~79% | ~79% |

The catch: **all expert weights must be loaded into memory** even though only 2 of 8 are used per token. MoE saves compute but not memory.

---

## 3. Training Cost Comparison

MoE models are more expensive to train due to:

- Load balancing losses to prevent expert collapse (all tokens routing to one expert)
- Communication overhead in distributed training (all-to-all routing across GPUs)
- Larger total parameter count requires more memory even during training

In practice, MoE models are trained on the same or more tokens as equivalent dense models to achieve the same quality — the compute efficiency gain is at inference, not training.

---

## 4. Quality at Equivalent Inference Cost

The central promise of MoE: **match a larger dense model's quality at a smaller dense model's inference cost**.

| Dense equivalent | MoE model | MoE active params | MoE quality (MMLU) | Dense quality (MMLU) |
|-----------------|-----------|------------------|--------------------|--------------------|
| LLaMA 2 70B | Mixtral 8x7B | ~13B | 79.9% | 69.8% |
| GPT-3.5 | Mixtral 8x22B | ~39B | ~77% | ~70% |

Mixtral 8x7B uses 5× fewer active parameters than LLaMA 2 70B while scoring ~10 points higher on MMLU.

---

## 5. When to Use Each

**Use a dense model when:**

- Serving on limited GPU memory (MoE requires loading all experts)
- Deploying on a single GPU or CPU (MoE routing adds complexity)
- Fine-tuning budget is small (fewer total parameters to update in dense)
- Latency is the primary constraint (dense models have simpler compute graphs)

**Use a MoE model when:**

- Throughput matters more than per-request latency
- Multiple GPUs are available (experts can be distributed across GPUs)
- Quality-per-inference-FLOP is the primary metric
- Total memory budget allows loading all experts

---

*Sources: Jiang et al. (2024) — Mixtral [[arXiv:2401.04088]](https://arxiv.org/abs/2401.04088) · DeepSeek-AI (2024) — DeepSeek-V2 [[arXiv:2405.04434]](https://arxiv.org/abs/2405.04434)*
