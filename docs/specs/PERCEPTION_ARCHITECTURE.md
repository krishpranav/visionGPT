# Perception Architecture

Version: 0.1

---

# Purpose

The Perception Engine transforms raw pixels into structured visual
representations suitable for downstream reasoning.

It does **not** answer questions.

It does **not** perform reasoning.

It only learns visual understanding.

---

# Pipeline

```text
Image
    │
    ▼
Patch Tokenizer
    │
    ▼
Position Embedding
    │
    ▼
Transformer Encoder
    │
    ▼
Multi-scale Features
    │
    ▼
Region Proposals
    │
    ▼
Object Features
```

---

# Transformer Design

Architecture:

```
Pre-Norm Transformer
```

Reason:

- Better optimization stability
- Better gradient flow
- Standard in modern large models

---

# Block Structure

```text
Input
 │
 ▼
LayerNorm
 │
 ▼
Self Attention
 │
 ▼
Residual
 │
 ▼
LayerNorm
 │
 ▼
MLP
 │
 ▼
Residual
```

---

# Activation

SwiGLU

Reason:

- Better scaling than GELU
- Used successfully in modern frontier models

---

# Attention

Current:

Global Multi-Head Self Attention

Future:

- FlashAttention
- Window Attention
- Sparse Attention

The interface must remain unchanged.

---

# Positional Encoding

Current:

Learnable Absolute Embeddings

Future:

2D Relative Position Bias

The encoder should not depend on the embedding implementation.

---

# Encoder Outputs

The encoder produces:

```text
Patch Features

↓

Object Discovery

↓

Scene Graph
```

It does not produce language.

---

# Design Principles

The Perception Engine owns:

- image understanding
- feature extraction
- visual representations

The Perception Engine never owns:

- reasoning
- language generation
- response formatting