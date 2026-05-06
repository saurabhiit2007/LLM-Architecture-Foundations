# Encoder, Decoder, and Encoder-Decoder Models

## 1. Encoder-Only Models (The "Understanders")
Encoder-only models focus on creating a rich, bidirectional representation of the input.

* **Mechanism:** Uses **Full (Bidirectional) Self-Attention**. Every token can attend to every other token in the sequence.
* **Mathematical Objective:** Typically **Masked Language Modeling (MLM)**.
    $$P(w_i | w_1, \dots, w_{i-1}, w_{i+1}, \dots, w_n)$$

* **Key Models:** BERT, RoBERTa, ALBERT.
* **When to Prefer:** When the task requires understanding the "whole-sentence" context simultaneously.
    * **Tasks:** Sentiment analysis, Named Entity Recognition (NER), and extractive Question Answering.

---

## 2. Decoder-Only Models (The "Generators")
These are the backbone of modern Generative AI. They are designed for one-way, autoregressive generation.

* **Mechanism:** Uses **Causal (Masked) Self-Attention**. A token can only attend to previous tokens in the sequence to prevent "cheating" by looking at the future.
* **Mathematical Objective:** **Causal Language Modeling (CLM)** or Next-Token Prediction.
    $$P(w_i | w_1, \dots, w_{i-1})$$

* **Key Models:** GPT-4, Llama 3/4, DeepSeek-R1, Mistral.
* **When to Prefer:** For open-ended generation, reasoning, and instruction following.
    * **Tasks:** Chatbots, creative writing, and code generation.

---

## 3. Encoder-Decoder Models (The "Translators")
The classic "Seq2Seq" architecture that combines an understanding phase with a generation phase.

* **Mechanism:** 
    1.  **Encoder:** Processes input bidirectionally.
    2.  **Decoder:** Generates output using causal attention + **Cross-Attention** (looking back at the Encoder's output).
* **Key Models:** T5, BART, FLAN-T5.
* **When to Prefer:** When the input and output are distinct but highly correlated sequences.
    * **Tasks:** Machine Translation (English to French) and Abstractive Summarization.

---

## Comparison Table

| Feature | Encoder-Only | Decoder-Only | Encoder-Decoder |
| :--- | :--- | :--- | :--- |
| **Attention** | Bidirectional | Causal | Bidirectional (Enc) + Causal (Dec) |
| **Primary Strength** | NLU (Understanding) | NLG (Generation) | Seq2Seq (Transformation) |
| **KV Cache Used** | No | Yes (Crucial for inference) | Yes (In the decoder) |
| **Scaling** | Diminishing returns | Scales best with compute | Stable for specific tasks |

---

## Updates (2026)

1.  **The Shift to Decoder-Only:** Interviewers may ask why BERT is "dead." The answer isn't that it's bad, but that Decoder-only models have shown superior **Scaling Laws**. By scaling a decoder, we get "emergent" understanding capabilities that rival encoders, making them more versatile.
2.  **Linear Attention:** Mention that standard $O(n^2)$ attention is being challenged by **Mamba (SSMs)** or **MLA (Multi-head Latent Attention)** used in DeepSeek models to handle massive contexts efficiently.
3.  **Thought Tokens:** With the rise of "reasoning" models (like o1 or R1), the decoder isn't just predicting the next word; it's generating a hidden "Chain of Thought."

---
