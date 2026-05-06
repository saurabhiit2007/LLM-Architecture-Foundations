# Normalization Techniques

## 1. Overview

Normalization is critical for training deep neural networks, especially transformers. It stabilizes training, speeds up convergence, and enables deeper architectures.

---

## 2. Layer Normalization (LayerNorm)

### What it does
Normalizes activations across the feature dimension for each sample independently.

### Formula
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

Where:

- `μ` = mean across features
- `σ²` = variance across features
- `γ`, `β` = learnable scale and shift parameters
- `ε` = small constant for numerical stability (typically 1e-5)

### Key Properties

- **Per-sample normalization**: Unlike BatchNorm, works on each sample independently
- **Feature-wise statistics**: Computes mean/variance across the feature dimension
- **No batch dependency**: Works with batch size = 1, crucial for inference
- **Commonly used in**: Transformers, RNNs

### Example
For input shape `[batch_size, seq_len, hidden_dim]`:

- Normalize across `hidden_dim` for each token independently
- Each token gets normalized using its own mean and variance

---

## 3. RMS Normalization (RMSNorm)

### What it does
Simplified version of LayerNorm that only uses root mean square for normalization, without mean centering.

### Formula

```
RMSNorm(x) = γ * x / RMS(x)
where RMS(x) = √(mean(x²) + ε)
```

### Key Differences from LayerNorm

- **No mean subtraction**: Doesn't center the data
- **No bias term**: Only has scale parameter γ (no β)
- **Faster**: ~15-20% faster than LayerNorm
- **Similar performance**: Empirically performs as well as LayerNorm in practice

### When to use

- Modern LLMs (LLaMA, GPT-J, etc.) prefer RMSNorm for efficiency
- When training speed matters and you don't need mean centering

---

## 4. Pre-LN vs Post-LN

The placement of normalization layers significantly impacts training stability and performance.

### Post-LN (Original Transformer)

**Architecture**:
```
x → MultiHeadAttention → Add → LayerNorm
       ↑___________________|
       
x → FeedForward → Add → LayerNorm
   ↑______________|
```

**Characteristics**:

- Normalization applied **after** the residual connection
- Original design in "Attention is All You Need"
- **Challenges**:
  - Harder to train deep models
  - Requires careful learning rate warmup
  - Gradients can become unstable

### Pre-LN (Modern Standard)

**Architecture**:
```
x → LayerNorm → MultiHeadAttention → Add
   ↑_____________________________|
   
x → LayerNorm → FeedForward → Add
   ↑______________________|
```

**Characteristics**:

- Normalization applied **before** the sublayer (attention/FFN)
- **Advantages**:
  - More stable training for deep networks
  - Can train without warmup
  - Better gradient flow
  - Easier to scale to deeper models
- **Widely adopted**: GPT-2, GPT-3, BERT variants, modern LLMs

### Comparison

| Aspect | Post-LN | Pre-LN |
|--------|---------|--------|
| Training Stability | Lower | Higher |
| Warmup Required | Yes | No |
| Deep Model Support | Harder | Easier |
| Original Paper | ✓ | ✗ |
| Modern Usage | Rare | Standard |

---

## 5. Practical Implementation Tips

### LayerNorm in PyTorch
```python
import torch.nn as nn

# Standard LayerNorm
layer_norm = nn.LayerNorm(hidden_dim)

# Pre-LN Transformer Block
class PreLNBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim)
    
    def forward(self, x):
        # Pre-LN: normalize before sublayer
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

### RMSNorm in PyTorch
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
```

---
