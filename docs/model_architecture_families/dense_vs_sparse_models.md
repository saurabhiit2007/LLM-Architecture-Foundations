## 1. High-Level Concept

The fundamental difference between Dense and Sparse models lies in the relationship between **Active Parameters** and **Total Parameters**.

- **Dense Models (e.g., LLaMA 3, GPT-3):**
  All model parameters are activated for every token. Increasing model capacity directly increases per-token compute and memory cost.


- **Sparse Models / Mixture of Experts (MoE) (e.g., Mixtral, some DeepSeek variants):**
  The model contains a large pool of parameters, but only a small subset is activated per token. This enables scaling model capacity faster than per-token compute, although system-level costs still increase.

> Note: MoE decouples total parameter count from per-token compute, but does not eliminate compute, memory, or system overhead.

---

---

## 2. Architecture Comparison

### 2.1 Dense Transformer Block

In a standard Transformer layer, each token passes through:

1. Self-Attention
2. A single dense Feed-Forward Network (FFN)

All FFN parameters are evaluated for every token.

---

### 2.2 Sparse (MoE) Transformer Block

MoE replaces the single FFN with:

1. **Experts:**
   A set of $N$ independent FFNs, commonly ranging from 8 to 64 experts.

2. **Router (Gating Network):**
   A lightweight network that selects which experts should process each token.

**Important architectural notes:**

- MoE typically replaces only the FFN, not the attention mechanism.
- Many architectures interleave dense and MoE layers rather than applying MoE everywhere.



**Mathematical Formulation**

For an input token $x$, the output is a weighted combination of the selected experts:

$$
y = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)
$$

Where:
- $G(x)_i$ is the routing probability for expert $i$
- $E_i(x)$ is the output of expert $i$
- Most models use **Top-2 routing during training** and often **Top-1 routing during inference**

> Note: Detailed description of MoE is available [here](./mixture_of_experts.md)

---

---

## 3. Dense vs. Sparse Comparison Table

| Feature | Dense Models | Sparse Models (MoE) |
|-------|--------------|---------------------|
| **Per-Token Compute** | Proportional to model size | Proportional to active experts |
| **Total Parameters** | Equal to active parameters | Much larger than active parameters |
| **Memory Footprint** | Proportional to total parameters | Proportional to total parameters plus routing overhead |
| **Training Complexity** | Stable and well understood | Complex due to routing and load balancing |
| **Inference Latency** | Compute-bound | Often communication-bound |
| **Scaling Behavior** | Performance scales with compute | Performance scales with total capacity |

> Note:
> - MoE reduces FLOPs per token but can increase system overhead.
> - Dense models are typically compute-bound, while MoE models are often bandwidth-bound.

---

---

## 4. Key Engineering Challenges

### 4.1 Load Balancing and Auxiliary Losses

**Problem:**

Routers tend to over-select a small subset of experts, leading to expert collapse and under-trained experts.

**Solution:**

Auxiliary losses encourage uniform expert utilization by penalizing skewed token distributions. These are often referred to as load balancing or importance losses.

Common failure modes:

- Expert collapse
- Token dropping due to expert capacity limits

---

### 4.2 Expert Specialization

- Experts rarely align with human-interpretable domains such as coding or specific languages.
- In practice, experts specialize in syntactic patterns, token statistics, or latent semantic clusters.
- Some experts may become partially redundant.

Specialization is emergent rather than explicitly supervised.

---

### 4.3 System Design and Communication Overhead

In distributed training or inference:

- Tokens are sharded across devices.
- Experts are distributed across GPUs or nodes.
- Tokens frequently require all-to-all communication to reach the selected experts.

> Therefore, Sparse models are often limited by network bandwidth and communication latency rather than raw FLOPs.

During inference, systems may reduce communication overhead using:

- Static expert placement
- Expert parallelism
- Reduced routing flexibility

---

---

## 5. Summary of Pros and Cons

### 5.1 Pros of Sparse (MoE)
- Higher model capacity at similar per-token compute
- Improved scaling efficiency at very large parameter counts
- Lower training compute for a given performance target

### 5.2 Cons of Sparse (MoE)
- Memory usage scales with total parameters, not active parameters
- Complex training dynamics and sensitivity to routing hyperparameters
- Higher system and serving complexity
- Inference latency can be dominated by communication overhead

**Example:**
A 47B MoE model with 12B active parameters still requires memory comparable to a 47B dense model.

---

---