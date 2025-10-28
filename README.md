# 2025-Artificial-Intelligence---Challenge-1
# Vietnamese Food Classification

Deep Learning image classification project for Vietnamese dishes using TensorFlow/Keras and Gradio.

## Objective

Build an image classification model for **3 popular Vietnamese dishes**:
- **Pho** (pho) - Vietnamese noodle soup
- **Com Tam** (com_tam) - Broken rice with grilled pork
- **Bun** (bun) - Vermicelli noodles

## Architecture

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Transfer Learning**: 2-phase training
  - Phase 1: Freeze base, train head (8 epochs, lr=1e-3)
  - Phase 2: Fine-tune last 20 layers (5 epochs, lr=1e-5)
- **Input**: 224x224 RGB images
- **Augmentation**: Random flip, rotation (±5°), zoom (±10%)

## Directory Structure

