# FreeU: Free Lunch in Diffusion U-Net

Advanced implementation and experimental evaluation of **FreeU**, a training-free enhancement technique for Stable Diffusion models.  
This project reimplements the original FreeU paper and introduces several novel optimizations, analysis tools, and engineering improvements for modern Diffusers pipelines.

---

## Overview

Diffusion models generate high-quality images through iterative denoising using a U-Net architecture.  
FreeU improves image quality during inference without requiring:

- Additional training
- Extra parameters
- Increased inference time
- Higher memory consumption

The method works by:
- Amplifying backbone feature maps for stronger semantic denoising
- Modulating skip connection frequencies using Fourier filtering

This project faithfully reproduces the original method and extends it with additional research contributions.

---

## Features

### Original FreeU Implementation
- Backbone feature scaling
- Fourier-domain skip feature modulation
- Stable Diffusion v1.5 integration

### Engineering Improvements
- Fixed Diffusers v0.27+ hook injection incompatibility
- Robust monkey-patching implementation for U-Net decoder blocks

### Novel Optimizations
- Timestep-adaptive backbone scaling
- Per-channel frequency thresholds
- Spectral logging for interpretability

### Experimental Analysis
- Quantitative evaluation across 10 prompts
- CLIP score analysis
- Tenengrad sharpness evaluation
- Local contrast measurement
- Spectral ratio analysis
- CFG-scale vs backbone-factor interaction study

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Diffusers
- Stable Diffusion v1.5
- OpenCV
- scikit-image
- NumPy
- Matplotlib

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Model | Stable Diffusion v1.5 |
| Scheduler | DPM-Solver++ |
| Inference Steps | 25 |
| GPU | NVIDIA T4 |
| Platform | Google Colab |

---

## Results

| Metric | Improvement |
|---|---|
| Tenengrad Sharpness | +17.3% |
| Local Contrast | +13.3% |
| Spectral Ratio | Improved |
| Additional Training Cost | None |

The experiments demonstrate that FreeU significantly improves local texture richness and image sharpness without modifying model weights.

---

## Novel Contributions

### 1. Adaptive Backbone Scaling
Dynamic backbone amplification based on denoising timestep progression.

### 2. Per-Channel Frequency Thresholds
Channel-aware spectral filtering for improved detail preservation.

### 3. Spectral Logging Infrastructure
Step-wise Fourier spectrum analysis for interpretability and debugging.

### 4. CFG × Backbone Interaction Study
A novel analysis showing that optimal FreeU hyperparameters depend on classifier-free guidance scale.

---

## Repository Structure

```bash
.
├── notebooks/
├── src/
├── evaluation/
├── results/
├── figures/
├── README.md
└── requirements.txt
