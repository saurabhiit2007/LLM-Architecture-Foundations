# WordPiece Tokenization

## 1. Overview

WordPiece is the subword tokenization algorithm used by BERT, RoBERTa, DistilBERT, and most Google models. Like BPE, it builds a vocabulary by iteratively merging character-level tokens — but the merge criterion is different.

---

## 2. How It Differs from BPE

| Aspect | BPE | WordPiece |
|--------|-----|-----------|
| Merge criterion | Highest **frequency** of pair | Highest **likelihood gain** of pair |
| Formula | $\text{count}(AB)$ | $\frac{P(AB)}{P(A) \cdot P(B)}$ |
| Effect | Merges the most common pairs first | Merges pairs that are most "surprising" together (not well-explained by parts alone) |
| Used in | GPT, LLaMA 3 (via TikToken) | BERT, RoBERTa, Google models |

The likelihood ratio $\frac{P(AB)}{P(A) \cdot P(B)}$ means WordPiece prefers merging tokens that co-occur much more than expected by chance — a statistical coherence criterion rather than raw count.

---

## 3. Subword Prefix Convention

WordPiece marks non-initial subwords with a `##` prefix:

```
"unhappiness" → ["un", "##happy", "##ness"]
"playing"     → ["play", "##ing"]
```

This convention makes it easy to reconstruct the original word by joining tokens and removing `##`. BPE and SentencePiece use a leading space (`▁`) to mark word boundaries instead.

---

## 4. Vocabulary

- **BERT base**: 30,522 tokens (English)
- **multilingual BERT**: 119,547 tokens (104 languages)
- Vocabulary is built on the training corpus; rare words fall back to character-level decomposition

---

*Source: Schuster & Nakamura (2012) — Japanese and Korean Voice Search (original WordPiece) · Devlin et al. (2019) — BERT [[arXiv:1810.04805]](https://arxiv.org/abs/1810.04805)*
