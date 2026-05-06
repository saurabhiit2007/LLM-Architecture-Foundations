# Unigram Tokenization

## 1. Overview

Unigram tokenization is a subword tokenization method that uses a probabilistic language model to segment text into tokens. Unlike BPE (bottom-up merging), Unigram starts with a large vocabulary and iteratively removes tokens to optimize the likelihood of training data.

---
## 2 How It Works?

### 2.1 Core Algorithm

1. **Initialize**: Start with a large seed vocabulary (characters + common substrings)
2. **Assign Probabilities**: Use EM algorithm to calculate token probabilities
3. **Prune**: Iteratively remove tokens that minimally impact overall likelihood
4. **Tokenize**: Use Viterbi algorithm to find most probable segmentation

### 2.2 Key Formula

For sentence x, find segmentation that maximizes:
```
P(x) = Π P(token_i)
```

---

## 3. Example Walkthrough

### Training Corpus
```
low, lower, newest, widest
```

### Initial Vocabulary with Probabilities
```
Token      | Probability
-----------|------------
low        | 0.15
est        | 0.13
lo         | 0.12
er         | 0.11
w          | 0.08
...
```

### Tokenizing "lowest"

**Possible segmentations:**

1. `low` + `est` → 0.15 × 0.13 = **0.0195** ✓ (Best)
2. `lo` + `w` + `est` → 0.12 × 0.08 × 0.13 = 0.001248
3. `l` + `o` + `w` + `e` + `s` + `t` → (very low probability)

**Selected**: `low` + `est` (highest probability)

## Comparison with Other Methods

| Feature | Unigram | BPE | WordPiece |
|---------|---------|-----|-----------|
| Approach | Top-down (prune) | Bottom-up (merge) | Bottom-up (merge) |
| Probabilistic | ✓ | ✗ | ✗ |
| Deterministic | ✗ | ✓ | ✓ |
| Training | Likelihood optimization | Frequency-based | Likelihood merging |
