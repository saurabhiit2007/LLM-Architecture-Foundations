# Next Token Prediction

## 1. Overview

Next token prediction (also called **causal language modeling** or **autoregressive language modeling**) is the pretraining objective used by all decoder-only LLMs including GPT, LLaMA, Mistral, and Claude. The model learns to predict the next token given all previous tokens in a sequence.

---

## 2. Training Objective

Given a sequence of tokens $x_1, x_2, \ldots, x_T$, the model maximizes the log-likelihood:

$$\mathcal{L} = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

In practice, this is minimized as cross-entropy loss:

$$\mathcal{L}_\text{CE} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

The model receives the full sequence as input and predicts each token from its prefix simultaneously (using a **causal mask** to prevent attending to future tokens). This makes pretraining highly parallelizable — all positions are trained in a single forward pass.

---

## 3. Causal Mask

The causal (autoregressive) mask ensures that position $t$ can only attend to positions $\leq t$:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

This mask is added to attention logits before softmax, making future tokens effectively invisible. The model is trained on all positions simultaneously but each position only "sees" its own past.

---

## 4. Teacher Forcing

During training, **teacher forcing** is used: the ground-truth tokens $x_1, \ldots, x_{t-1}$ are always fed as input, not the model's own previous predictions. This:

- Makes training stable and fully parallelizable
- Avoids compounding errors during training
- Creates an **exposure bias**: at inference time, the model conditions on its own (potentially incorrect) previous outputs, which it never saw during training

Exposure bias is a known limitation of teacher forcing but has not prevented LLMs from achieving strong generation quality in practice.

---

## 5. Perplexity

**Perplexity (PPL)** is the standard metric for language model quality during pretraining:

$$\text{PPL} = \exp\!\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\right)$$

Intuitively: if perplexity is $k$, the model is as uncertain as if it were choosing uniformly among $k$ options at each step. Lower is better.

- GPT-2 (1.5B) on WikiText-103: PPL ≈ 18
- GPT-3 (175B) on Penn Treebank: PPL ≈ 20
- LLaMA 2 (70B): PPL ≈ 3.3 on the Pile

Perplexity is corpus-dependent and cannot be compared directly across different tokenizers or evaluation sets.

---

## 6. Why Next Token Prediction Works

The simplicity of the objective is intentional. By predicting the next token, the model must implicitly learn:

- Grammar and syntax (to predict grammatically correct continuations)
- World knowledge (to predict factually plausible continuations)
- Reasoning patterns (to predict conclusions that follow from premises)
- Code structure (to predict syntactically and semantically valid code)

None of these are explicitly supervised — they emerge as instrumental goals in service of next-token prediction accuracy.

---

## 7. Pretraining vs Other Objectives

| Objective | Used by | Attention | Strength |
|-----------|---------|-----------|----------|
| Next token prediction (CLM) | GPT, LLaMA, Mistral | Causal (decoder-only) | Generation, few-shot |
| Masked language modeling (MLM) | BERT, RoBERTa | Bidirectional (encoder) | Classification, NLU |
| Span corruption (T5) | T5, mT5 | Encoder-decoder | Seq2seq tasks |

Decoder-only NTP has become dominant for generalist LLMs because it generalizes well to both generation and understanding tasks when scaled.

---

*Sources: Radford et al. (2019) — GPT-2 [[openai.com]](https://openai.com/index/better-language-models/) · Brown et al. (2020) — GPT-3 [[arXiv:2005.14165]](https://arxiv.org/abs/2005.14165)*
