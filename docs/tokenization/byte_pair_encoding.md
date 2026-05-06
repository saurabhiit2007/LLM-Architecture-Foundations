# Byte Pair Encoding (BPE)

## 1. Overview

Byte Pair Encoding (BPE) and its modern variants are the dominant subword tokenization methods used in today’s Large Language Models (LLMs), including **GPT-3, GPT-4, and LLaMA**. BPE strikes a practical balance between word-level and character-level tokenization, enabling scalable training while avoiding out-of-vocabulary failures.

BPE addresses the **out-of-vocabulary (OOV)** problem by decomposing text into frequently occurring subword units instead of relying on a fixed word vocabulary. Any unseen word can still be represented as a sequence of known subcomponents.

This makes BPE robust, flexible, and suitable for web-scale corpora with noisy and evolving language.

---

## 2. How BPE Works (The Algorithm)

1. **Initialize**  
   Start with a vocabulary of base symbols, either characters or UTF-8 bytes.

2. **Frequency Analysis**  
   Count all adjacent token pairs in the training corpus.

3. **Merge**  
   Replace the most frequent adjacent pair with a new token.

4. **Iterate**  
   Repeat until a predefined vocabulary size is reached.

### 2.1 Step-by-Step Example

Consider a tiny corpus: [low, lower, newest, widest]

#### Step 1: Initialize

Start with characters as the base vocabulary and mark word boundaries:

> l o w _ <br>
> l o w e r _ <br>
> n e w e s t _ <br>
> w i d e s t _ <br>

Initial tokens are individual characters.

---

#### Step 2: Frequency Analysis

Count all adjacent token pairs across the corpus.

Some frequent pairs:

- `l o`
- `o w`
- `e s`
- `s t`

Assume `o w` is the most frequent pair.

---

#### Step 3: Merge

Merge `o w → ow` everywhere:

> l ow _ <br>
> l ow e r _ <br>
> n e w e s t _ <br>
> w i d e s t _ <br>

Vocabulary now includes `ow`.

---

#### Step 4: Iterate

Repeat frequency analysis and merging.

Next merges might be:

- `l ow → low`
- `e s → es`
- `es t → est`

Eventually the corpus may look like:

> low _ <br>
> lower _ <br>
> new est _ <br>
> wid est _ <br>

And the vocabulary contains: [l, o, w, e, r, n, i, d, t, low, est, wid, lower, _]

The process stops once the target vocabulary size is reached.

---

### 2.2 Vocabulary Size

The final vocabulary size is:

$$
V = S + N_{\text{merges}}
$$

Where:

- $S$ is the number of base symbols
- $N_{\text{merges}}$ is the number of merge operations

Each merge permanently adds one new token.

---

### 2.3 Key Properties

- **Greedy:** Always merges the most frequent pair
- **Deterministic:** Same data and merges produce the same vocabulary
- **Irreversible:** Bad early merges cannot be undone

---

### 2.4 Where BPE Shines

#### 1. Common Words and Morphemes

Frequently occurring patterns like:

- `ing`, `tion`, `http`, `://`
become single tokens, reducing sequence length and improving efficiency.

#### 2. Open Vocabulary

Any new word can be represented as subwords: 

`unhappiness → un + happi + ness`

No `[UNK]` tokens are required.

#### 3. Web-Scale Text

BPE handles:

- URLs
- Code
- Typos
- Mixed-language text

very well, especially in byte-level variants.


### 2.5 Where BPE Fails or Struggles

#### 1. Rare Words and Low-Resource Languages

Languages with rich morphology or limited data are often split into many tokens, making them:

- More expensive to process
- Harder to learn effectively

#### 2. Numbers and Arithmetic

BPE tokenizes numbers inconsistently:

`1000 → [1000]`
`10001 → [100, 01]`

This breaks digit-level reasoning and arithmetic.

#### 3. Spelling and Character-Level Tasks

If a word becomes a single token:
`strawberry → [strawberry]`

The model cannot directly reason about individual letters.

#### 4. Early Merge Bias

Because merges are greedy:

- Early frequent patterns dominate the vocabulary
- Suboptimal merges persist forever
- Later data distributions cannot correct them

#### 5. Sensitivity to Formatting

Whitespace, casing, and punctuation affect token boundaries: `"hello" ≠ " hello"`

Prompt formatting can significantly change tokenization.

---

## 3. Tokenization Strategies Compared

| Strategy | Pros | Cons |
|--------|------|------|
| Word-level | High semantic meaning | Huge vocabulary, OOV issues |
| Character-level | No OOV, small vocabulary | Long sequences, weak semantics |
| Subword (BPE) | Balanced efficiency and flexibility | Linguistically unintuitive splits |

---

## 4. Why BPE Works Well for LLMs

- **Token Efficiency**  
  Frequent strings like `the`, `ing`, or `http` become single tokens, reducing sequence length.

- **Statistical Morphology**  
  Related words such as `play`, `playing`, and `played` often share subword units.  
  This is an emergent frequency effect, not explicit linguistic understanding.

- **Byte-level Robustness**  
  Byte-level BPE operates on UTF-8 bytes, guaranteeing that any string can be tokenized without an `[UNK]` token.

---

## 5. BPE Variants Used in Practice

Modern LLMs rarely use vanilla BPE.

- **Byte-level BPE (GPT-2, GPT-4)**  
  Uses bytes as base symbols, eliminating unknown tokens.

- **SentencePiece BPE (LLaMA)**  
  Avoids pre-tokenization and treats whitespace as a normal symbol.

- **Unigram Language Model (SentencePiece)**  
  Uses a probabilistic model over subwords instead of greedy merges, allowing multiple valid tokenizations.

Key distinction:

- BPE is deterministic and greedy
- Unigram LM is probabilistic and more flexible

---

## 6. Pre-tokenization and Whitespace Effects

Tokenization is sensitive to whitespace and formatting.

- `"hello"` and `" hello"` may map to different tokens
- Leading spaces often act as token boundaries
- Prompt formatting directly influences tokenization

These effects explain why small formatting changes can significantly alter model behavior.

---

## 7. Tokenization Paradoxes (Interview-Level Insights)

### The Spelling Paradox
LLMs struggle to count letters in words.

- Reason: words often appear as single tokens, hiding character-level structure.

### The Arithmetic Paradox
LLMs fail at digit-wise arithmetic on large numbers.

- Reason: numbers are split inconsistently across tokens.

### The Case Sensitivity Trap
`Hello`, `hello`, and `HELLO` are distinct tokens.

- Effect: the model must learn redundant representations.

---

## 9. Tokenization Is Part of the Model

Tokenization defines:

- The atomic prediction units
- The embedding space structure
- What patterns are easy or hard to learn

Changing tokenization usually requires:

- Relearning embeddings
- Often retraining the entire model

Tokenization errors cannot be fixed by fine-tuning alone.

---
