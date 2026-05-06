# TikToken

## 1. Overview

`tiktoken` is a high-performance **Byte Pair Encoding (BPE)** tokenizer developed by OpenAI. It is written in **Rust**, making it significantly faster than previous Python-based tokenizers. It is the backbone for how GPT models process text.

---

## 2. Core Encodings & Model Mapping

| Encoding | Models | Key Characteristics |
| :--- | :--- | :--- |
| **o200k_base** | GPT-4o, GPT-4o-mini | Largest vocab (~200k); best for non-English languages. |
| **cl100k_base** | GPT-4, GPT-3.5-Turbo | Standard for the "Chat" era; ~100k vocab. |
| **p50k_base** | GPT-3, Codex | Legacy; used for older completion models. |

---

## 3. Why `tiktoken`?

### A. Performance & Speed
Because it is implemented in Rust, it can handle massive datasets much faster than the `transformers` library's Python implementation. This is critical for real-time applications.

### B. Efficiency (The GPT-4o Update)
The recent shift to **o200k_base** is a major milestone.

* **Token Compression:** It packs more text into fewer tokens.
* **Multilingual Support:** Older tokenizers were "English-centric." The new version reduces costs for users writing in languages like Hindi, Japanese, or Arabic by reducing the token count for the same meaning.

### C. Visualizing the Process
The BPE algorithm starts with individual bytes and iteratively merges the most frequent adjacent pairs.

---

## 4. Technical Implementation (Python)

```python
import tiktoken

# Load the specific encoding for GPT-4o
enc = tiktoken.get_encoding("o200k_base")

# Encode text
text = "Tokenization is essential."
tokens = enc.encode(text)
print(f"Tokens: {tokens}")

# Decode back to text
original = enc.decode(tokens)
print(f"Decoded: {original}")

# Check vocabulary size
print(f"Vocab size: {enc.n_vocab}")
```

---
