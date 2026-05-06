# Scaling Laws and Compute-Optimal Training

## 1. Overview

Scaling laws describe how model performance (measured by loss) changes as a function of three variables: **model size** ($N$, parameters), **dataset size** ($D$, tokens), and **compute budget** ($C$, FLOPs). Two landmark papers established the empirical relationship:

- **Kaplan et al. (2020)** — OpenAI: first systematic scaling laws for LLMs
- **Hoffmann et al. (2022)** — DeepMind: Chinchilla scaling laws (revised optimal allocation)

---

## 2. Kaplan et al. (2020): Power Laws

Kaplan et al. found that LLM loss follows clean **power laws** in each variable when the others are held fixed:

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

With approximately: $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$

**Key finding:** Model size ($N$) has a stronger effect on loss than dataset size ($D$). Kaplan et al. concluded that given a fixed compute budget, it is better to **train a larger model on fewer tokens** than a smaller model to convergence.

---

## 3. Chinchilla (Hoffmann et al., 2022): The Revised Optimum

Chinchilla challenged Kaplan's recommendation by training models more carefully across a wider range of sizes and token counts. Their finding:

**For a compute-optimal run, model size $N$ and training tokens $D$ should scale equally:**

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

In practice: **~20 tokens per parameter** is the compute-optimal ratio.

| Model | Parameters | Training tokens | Tokens/param | Chinchilla optimal? |
|-------|-----------|----------------|-------------|-------------------|
| GPT-3 | 175B | 300B | 1.7× | Under-trained |
| Gopher | 280B | 300B | 1.1× | Under-trained |
| Chinchilla | 70B | 1.4T | 20× | Yes (by design) |
| LLaMA 2 7B | 7B | 2T | 286× | Over-trained |
| LLaMA 3 8B | 8B | 15T | 1875× | Far over-trained |

**The "over-training" insight:** Training a smaller model on far more tokens than Chinchilla-optimal is rational for **inference** — a smaller model trained on more data matches the quality of a larger Chinchilla-optimal model but is cheaper to serve. LLaMA 2/3 explicitly chose this trade-off.

---

## 4. Practical Implications

### Model size vs data size
- Given a fixed compute budget, Chinchilla says split compute equally between model size and data
- Given a fixed inference budget (serving cost), train smaller models on more data

### Why frontier models exceed Chinchilla
GPT-4, LLaMA 3, and others train on far more tokens than the 20× rule because:
1. Inference is run billions of times — even small quality gains at the cost of extra training compute are worthwhile
2. Data quality matters more than Chinchilla assumed; better-curated datasets push the optimum further

### Emergent capabilities and scaling
Some capabilities appear to emerge discontinuously at scale thresholds (Wei et al., 2022):
- Chain-of-thought reasoning: largely absent below ~100B parameters
- Arithmetic, multi-step reasoning: emerge at scale

However, later work (Schaeffer et al., 2023) argues some apparent emergent capabilities are artifacts of nonlinear evaluation metrics rather than true discontinuities.

---

## 5. The Scaling Hypothesis

The broader **scaling hypothesis** (Sutton's "Bitter Lesson", 2019) holds that general methods that leverage compute scale better than methods encoding human knowledge:

- More compute + more data + larger model → better performance, reliably
- Architecture search, feature engineering, and inductive biases have diminishing value as scale increases

This underpins why the field has converged on similar decoder-only Transformer architectures rather than divergent specialized designs.

---

*Sources: Kaplan et al. (2020) [[arXiv:2001.08361]](https://arxiv.org/abs/2001.08361) · Hoffmann et al. (2022) — Chinchilla [[arXiv:2203.15556]](https://arxiv.org/abs/2203.15556) · Wei et al. (2022) — Emergent Abilities [[arXiv:2206.07682]](https://arxiv.org/abs/2206.07682)*
