# Sequence Modeling From Scratch

Implementations of core sequence modeling architectures — built from the ground up using only NumPy. No PyTorch, no TensorFlow. Every forward pass, every gradient, every weight update written by hand.

This is not a tutorial follow-along. The goal was to deeply understand *why* these architectures work — by deriving and implementing the math myself.

---

## What's implemented

### Recurrent Networks
- **Vanilla RNN** — forward pass, BPTT (backpropagation through time), gradient clipping
- **LSTM** — all four gates, cell state, full backprop through gates
- **GRU** — reset and update gates, full backprop

### Transformer
- Multi-head self-attention (scaled dot-product)
- Positional encoding
- Feed-forward sublayers
- Layer normalization
- Encoder stack
- Forward pass fully implemented; backprop partially derived (in progress)

### Optimizers
- SGD
- Adam (with bias correction)

### Fundamentals
- Backpropagation basics
- Computational graph intuition
- Gradient flow through common operations

---

## Why from scratch?

Using a framework like PyTorch abstracts away what's actually happening during training. I wanted to understand:

- How gradients flow through an LSTM gate
- Why vanishing gradients happen in vanilla RNNs and how LSTMs fix it
- What attention is actually computing geometrically
- How Adam's bias correction works and why it matters early in training

Building these by hand forced answers to all of those questions.

---

## Structure

```
sequence-modeling-from-scratch/
├── fundamentals/     # Backprop basics, gradient flow
├── recurrent/        # Vanilla RNN
├── lstms/            # LSTM implementation
├── grus/             # GRU implementation
├── transformer/      # Transformer (attention, encoder, positional encoding)
├── optimizers/       # SGD, Adam
└── tests/            # Correctness checks
```

---

## What's next

- Complete transformer backprop
- Implement GPT-1 style decoder (autoregressive language model)
- Train on character-level text data and benchmark RNN vs LSTM vs Transformer

---

## Dependencies

```
numpy
```

That's it.