# Vocabulary Size Trade-offs

## 1. Overview

Vocabulary size — the number of unique tokens the tokenizer recognizes — is a fundamental design choice that affects model capacity, training efficiency, inference cost, and cross-lingual performance. Modern LLMs use vocabularies ranging from ~32K (LLaMA 2) to ~256K (Gemini).

---

## 2. The Core Trade-off

| Effect | Small vocabulary (~32K) | Large vocabulary (~100K+) |
|--------|------------------------|--------------------------|
| Avg. tokens per word | More (longer sequences) | Fewer (shorter sequences) |
| Sequence length | Longer → slower attention | Shorter → faster attention |
| Embedding table size | Small | Large (embedding + unembedding) |
| Rare word coverage | Poor (falls back to characters) | Good |
| Multilingual coverage | Limited | Better across scripts |
| OOV handling | More fragmented | Less fragmented |

---

## 3. Sequence Length Impact

A smaller vocabulary forces the tokenizer to split words into more pieces. This directly increases sequence length, which increases the quadratic cost of attention:

- GPT-2 (50K vocab): "unhappiness" → 3 tokens
- LLaMA 2 (32K vocab): "unhappiness" → 4 tokens
- GPT-4 (100K vocab): "unhappiness" → 1 token

Longer sequences mean more KV cache memory, slower prefill, and more forward passes per document.

---

## 4. Embedding Table Cost

Vocabulary size directly determines the size of two weight matrices:

- **Token embedding matrix**: $V \times d_\text{model}$
- **Unembedding (lm_head) matrix**: $d_\text{model} \times V$

For a model with $d_\text{model} = 4096$ and $V = 100{,}000$: each matrix is $4096 \times 100{,}000 \approx 400M$ parameters (in float32). This is a meaningful fraction of a 7B model's total parameters.

In practice, the two matrices are often **weight-tied** (shared weights) to halve this cost.

---

## 5. Multilingual and Script Coverage

Small vocabularies trained primarily on English text tokenize non-Latin scripts (Chinese, Arabic, Hindi) very inefficiently — each character may become its own token, producing extremely long sequences. Large multilingual vocabularies (SentencePiece with 250K+) allocate tokens proportionally to corpus frequency across languages.

**LLaMA 2 (32K, English-heavy):** A Hindi sentence of 10 words may expand to 40+ tokens.

**LLaMA 3 (128K):** Meta increased vocabulary size specifically to improve multilingual tokenization efficiency.

---

## 6. Fertility

**Fertility** is the average number of tokens produced per word for a given tokenizer on a given corpus. Lower fertility = more efficient tokenization.

- English text on a well-matched tokenizer: fertility ≈ 1.3–1.5
- Non-Latin script on a mismatched tokenizer: fertility can exceed 5–10

Fertility is a standard metric for evaluating multilingual tokenizer quality.

---

## 7. Real-World Vocabulary Sizes

| Model | Vocab Size | Notes |
|-------|-----------|-------|
| GPT-2 | 50,257 | BPE, English-focused |
| LLaMA 2 | 32,000 | SentencePiece BPE |
| LLaMA 3 | 128,256 | Enlarged for multilingual |
| GPT-4 / GPT-3.5 | ~100,000 | cl100k_base (TikToken) |
| Gemini | 256,000 | SentencePiece, multilingual |
| Mistral 7B | 32,000 | Same as LLaMA 2 |
| BLOOM | 250,880 | Multilingual, 46 languages |

---

## 8. Choosing Vocabulary Size

- **Monolingual English, smaller model**: 32K–50K is sufficient; keeps embedding table small
- **Multilingual or code-heavy**: 100K+ to avoid tokenization inefficiency
- **Very long context tasks**: Larger vocabulary reduces sequence length, reducing attention cost
- **Tight memory budget**: Smaller vocabulary reduces embedding table size

---

*Sources: Kudo & Richardson (2018) — SentencePiece [[arXiv:1808.06226]](https://arxiv.org/abs/1808.06226) · Touvron et al. (2023) — LLaMA 2 [[arXiv:2307.09288]](https://arxiv.org/abs/2307.09288) · Dubey et al. (2024) — LLaMA 3 [[arXiv:2407.21783]](https://arxiv.org/abs/2407.21783)*
