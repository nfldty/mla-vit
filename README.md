# MLA-ViT: Enhancing Vision Transformers with Multi-head Latent Attention

This repository contains the code and experiments for **MLA-ViT**, a Vision Transformer architecture that replaces standard Multi-Head Attention (MHA) with **Multi-Head Latent Attention (MLA)** to improve efficiency.

##  Overview

Vision Transformers (ViTs) are powerful but often limited by their quadratic memory and compute complexity in attention layers. Our model, **MLA-ViT**, integrates low-rank latent projections and Rotary Positional Embeddings (RoPE) to reduce training and inference costs without sacrificing accuracy.

> **Key Results (CIFAR-10):**
> -  ~23% faster training time
> -  ~29% lower memory usage
> -  Comparable accuracy to baseline ViT

##  Experimental Setup

- **Dataset:** CIFAR-10 (32x32, 10 classes)
- **Training Epochs:** 300
- **Optimizer:** Adam, lr=1e-3
- **Batch Size:** 1000
- **Hardware:** Single NVIDIA GPU
- **Patch Size:** 4x4
