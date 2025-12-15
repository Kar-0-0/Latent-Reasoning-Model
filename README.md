# Latent Reasoning Model

A hybrid sequence model combining recurrent-style latent dynamics with a Transformer decoder for long-range sequence reasoning.  

This project explores **long-sequence learning** using a gated latent state (`LatentDynamics`) that mixes decayed past information with input-dependent updates, followed by attention-based decoding. The goal is to handle sequences longer than what standard Transformers or small recurrent models can efficiently manage.

---

## Table of Contents

- [Overview](#overview)  
- [Model Architecture](#model-architecture)  
- [Datasets](#datasets)  
- [Training](#training)  
- [Results So Far](#results-so-far)  
- [Future Directions](#future-directions)  
- [Requirements](#requirements)  

---

## Overview

The Latent Reasoning Model is a **custom hybrid sequence model** designed to capture long-term dependencies in sequences. It is composed of:  

1. **LatentEncoder** – maps discrete tokens into a learnable latent space.  
2. **LatentDynamics** – gated recurrent-like updates with a learnable per-dimension “leak” factor α, inspired by GRU/LSTM dynamics.  
3. **LatentDecoder** – a stack of Transformer-like blocks that attend over latent states in chunks for memory efficiency.  

This design allows handling sequences longer than standard Transformers would fit into GPU memory, while still leveraging attention for decoding.

---

## Model Architecture

- **LatentEncoder**: `Embedding → Linear` to produce latent states `z`.  
- **LatentDynamics**: recurrence with learnable leak α, updated per timestep:  

```cpp
h_t = tanh((1 - α) * h_{t-1} + gate(h_{t-1}, z_t) ⊙ z_t)
```

- **LatentDecoder**: multiple Transformer-style blocks with attention and feedforward layers applied to latent states.  
- **Chunked Attention**: attention is computed in manageable chunks to reduce GPU memory usage for long sequences.  

---

## Datasets

Custom datasets used to evaluate long-sequence reasoning:

1. **LongIncrementDataset**  
   - Sequences increment by 1, with the last token skipping by 5.  
   - Designed to test long-range sequential pattern learning.

2. **HiddenRuleSwitchDataset**  
   - Sequences that follow one rule for the first portion and switch to another rule mid-sequence.  
   - Tests model’s ability to adapt to changing rules.

3. **StructuredTokenDataset**  
   - Deterministic modulo-based sequences.  
   - Tests memorization and next-token prediction.  

---

## Training

- Implemented in PyTorch with **MPS GPU support**.  
- Supports **chunked sequence processing** to handle sequences > 500 tokens.  
- Standard cross-entropy loss for next-token prediction.  
- Example hyperparameters (can be scaled based on GPU memory):

```python
latent_dim = 64
attn_dim = 64
n_layers = 2
n_heads = 4
dropout = 0.1
batch_size = 32
chunk_size = 128
epochs = 20
```

- Latent states can be saved for analysis:

```python
all_latents = np.stack(latents_chunks)
np.save("latent_trajectories.npy", all_latents)
```

## Results So Far

- Successfully learns sequences of length up to ~500 tokens.
- Loss decreases reliably on short sequences, but longer sequences (>500 tokens) show “confidently wrong” predictions due to latent capacity limitations.
- Memory-efficient chunked attention allows longer sequences on MPS GPUs.

## Future Directions

1. **Scale latent capacity**  
   - Increase `latent_dim`, `attn_dim`, and number of layers to handle longer sequences.

2. **Positional encoding / causal masking**  
   - Add positional encodings or causal masks to the attention blocks to improve long-range order awareness.

3. **Curriculum learning**  
   - Train on gradually longer sequences to improve generalization.

4. **Benchmark against baselines**  
   - Compare against GRU + Transformer and pure Transformer architectures.

5. **Ablation studies**  
   - Test the effect of the learnable leak α and gating mechanism to isolate what contributes most to performance.

6. **Real-world datasets**  
   - Explore tasks like Tiny Shakespeare or algorithmic sequence tasks to demonstrate practical utility.

## Requirements

- Python 3.11+
- PyTorch with MPS support (or CUDA for other GPUs)
- NumPy
- Optional: tqdm for progress bars

```bash
pip install torch torchvision torchaudio numpy
```
