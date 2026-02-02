## 1. The Claude Model Family (Current Landscape)

Anthropic uses a "Tier" system (Haiku, Sonnet, Opus) across generations. As of 2026, the **4.5 Generation** is the state-of-the-art.

| Model | Generation | Tier | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Claude 4.5 Opus** | Frontier | System 2 | Extreme reasoning, scientific discovery, autonomous agents. |
| **Claude 4.5 Sonnet** | Mid-tier | Hybrid | Software engineering (Claude Code), enterprise workflows. |
| **Claude 4.5 Haiku** | Lightweight | System 1 | High-speed API tasks, sub-second latency requirements. |

> **Note:** Claude 3.7 Sonnet was the landmark "Hybrid" model that first introduced a toggle between instant (System 1) and extended thinking (System 2) modes.

---

---

## 2. Model Tiers & Their Cognitive Roles

Anthropic maintains three tiers (Haiku, Sonnet, Opus). In the 2025–2026 era, their roles have shifted toward specific agentic behaviors.

### **Tier 1: Haiku (The "System 1" Specialist)**
* **Mode:** Primarily System 1.
* **Role:** High-speed, high-volume tasks.
* **Context:** Used as the "Router" or "Small Brain" in agentic workflows to handle simple classification or fast sub-tasks.

### **Tier 2: Sonnet (The "Hybrid/Agentic" Workhorse)**
* **Mode:** Hybrid.
* **Role:** This is Anthropic's flagship tier for **Agents**. 
* **Key Feature:** **Computer Use.** Sonnet 3.5 and 4.5 are specifically tuned to use tools, navigate UIs, and write code via the `Claude Code` CLI.

### **Tier 3: Opus (The "System 2" Frontier)**
* **Mode:** Deep System 2.
* **Role:** Reserved for tasks where accuracy is more important than speed (e.g., drug discovery, complex legal analysis). 
* **Context:**  **Inference-Time Scaling Laws** - the idea that more "thinking time" leads to higher IQ.

---

---

## 3. The "Agentic" Evolution

While "System 2" refers to **how the model thinks**, "Agentic" refers to **what the model can do.**

* **Claude 3.5:** Introduced the "Agentic Framework" (Computer Use). It was a System 1 model trying to do System 2 tasks via tool-calling.
* **Claude 3.7:** Improved Agentic reliability by allowing the model to "think" (System 2) before it clicks a button or writes a line of code.
* **Claude 4.5:** Fully **Agentic-Native**. The model doesn't just call tools; it plans the entire multi-step trajectory using hidden thinking tokens before execution.
    - **Claude Code:** A specialized agentic interface that uses these capabilities to perform autonomous software engineering tasks (e.g., "Fix the bug in the auth flow and run the tests").

---

---

## 4. Core Differentiator: Constitutional AI (CAI)

Anthropic’s defining strategy is **Constitutional AI**, which enables **RLAIF** (Reinforcement Learning from AI Feedback).

### The Training Pipeline
1. **The Constitution:** A set of rules (e.g., "Be helpful, honest, and harmless") used as a rubric.
2. **Critique and Revision:** The model generates an initial response, then critiques itself based on the Constitution, and finally rewrites it.
3. **Preference Model (RLAIF):** A separate AI model evaluates thousands of these pairs to create a "Reward Model." 
4. **Reinforcement Learning:** The final model is fine-tuned to maximize the rewards defined by that AI-driven preference model.

> **Insight:** This reduces **Sycophancy** (the tendency of models to agree with users just to be "likable") because the model is optimized for a fixed Constitution rather than fickle human ratings.

---

---

## 5. System 1 vs. System 2 Reasoning

Anthropic has moved beyond "prompting for step-by-step" into true **Inference-Time Scaling**.

### System 1: Fast/Intuitive
- **Mechanism:** Standard next-token prediction.
- **Latency:** Instant.
- **Use Case:** Brainstorming, simple chat, data extraction.

### System 2: Deliberate/Extended Thinking
- **Mechanism:** The model generates **Thinking Tokens** (visible or hidden) before producing the final answer. This allows the model to "plan" and "search" through different solution paths.
- **Scaling Law:** Performance improves as you increase the **Thinking Budget** (more compute at inference = higher intelligence).
- **Models:** Claude 3.7 Sonnet and Claude 4.5 Opus.

---

---

## 6. Summary Cheat Sheet for Interviews

| Feature | Technical Detail to Mention |
| :--- | :--- |
| **Alignment** | Constitutional AI / RLAIF (Scalable oversight). |
| **Reasoning** | Inference-time Scaling (System 2 thinking tokens). |
| **Architecture** | Likely Mixture of Experts (MoE) to balance speed and capacity. |
| **Agentic** | Computer Use API (Screenshots to x,y coordinate actions). |
| **Safety** | Focused on "Red Teaming" and "ASL" (AI Safety Levels). |

---