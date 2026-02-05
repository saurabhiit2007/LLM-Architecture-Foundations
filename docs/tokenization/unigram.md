## 1. Overview

Unigram tokenization is a subword tokenization method that uses a probabilistic language model to segment text into tokens. Unlike BPE (bottom-up merging), Unigram starts with a large vocabulary and iteratively removes tokens to optimize the likelihood of training data.

---

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

## Interview Questions & Answers

### Q1: What is the main difference between Unigram and BPE?

**Answer:** BPE builds vocabulary bottom-up by merging frequent character pairs (deterministic), while Unigram starts with a large vocabulary and prunes it top-down based on likelihood optimization (probabilistic). Unigram can produce multiple segmentations for the same word, while BPE always produces the same segmentation.

---

### Q2: How does Unigram handle unseen words?

**Answer:** Unigram breaks unseen words into known subword units using the most probable segmentation. It includes individual characters as fallback, ensuring no out-of-vocabulary (OOV) words. For example, "unbreakable" might become "un" + "break" + "able" or fallback to character-level if needed.

---

### Q3: Explain the role of the EM algorithm in Unigram.

**Answer:** The EM algorithm estimates token probabilities:
- **E-step**: Calculate expected counts of each token across all possible segmentations
- **M-step**: Update token probabilities based on expected counts
- **Iterate**: Repeat until convergence

This ensures probabilities reflect each token's importance in the training corpus.

---

### Q4: What is the Viterbi algorithm's role?

**Answer:** Viterbi finds the most probable segmentation efficiently using dynamic programming. Instead of exploring all exponential segmentations (2^n), it computes the optimal path in O(n²) time, where n is the word length.

---

### Q5: How do you determine vocabulary size?

**Answer:** Vocabulary size is a hyperparameter (typically 8K-32K tokens) determined by:
- Task requirements and language complexity
- Corpus size and diversity
- Model size and computational constraints
- Downstream task performance

Start with a large vocabulary (10x target) and prune to desired size.

---

### Q6: What are Unigram's advantages?

**Answer:**
- Theoretically principled (optimizes clear objective)
- Handles ambiguity through probabilistic approach
- Better for morphologically rich languages
- Language and script agnostic
- No pre-tokenization required
- Robust to noisy input (typos, code-switching)

---

### Q7: How does vocabulary pruning work?

**Answer:** 
```
For each token:
  1. Calculate loss increase if removed
  2. Rank by loss impact
  3. Remove bottom 10-20% (least impact)
  4. Recalculate probabilities
  5. Repeat until target size
```

Tokens are redundant if other combinations can represent the same text with minimal probability loss.

---

### Q8: When would you prefer Unigram over BPE?

**Answer:** Prefer Unigram for:
- **Multilingual models** (handles 100+ languages without pre-processing)
- **Morphologically rich languages** (Finnish, Turkish, Japanese)
- **Noisy text** (social media, user-generated content)
- **Mixed scripts** (e.g., "I love 寿司")

Used in: Google NMT, T5, XLM-R, mBART

---

### Q9: How does Unigram handle multilingual scenarios?

**Answer:** Unigram is language-agnostic—it treats all input as raw sequences without requiring language identification. When trained on multilingual data, it learns subword units across languages with probabilities reflecting their importance. Naturally handles code-switching since it doesn't depend on word boundaries.

**Example**: "I love 寿司 and pasta" → ["I", "love", "寿", "司", "and", "pasta"]

---
