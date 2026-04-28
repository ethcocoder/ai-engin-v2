# AetherCodec-Elite: Codebase Analysis & Developer Documentation

## 1. Project Overview
**AetherCodec-Elite** is a state-of-the-art Learned Image Compression (LIC) system. Unlike traditional codecs (JPEG, PNG) which use hand-crafted algorithms, AetherCodec uses deep neural networks to learn the most efficient way to represent visual data. It optimizes for both high compression ratios and perceptual quality using a combination of Swin Transformers, GANs, and advanced entropy modeling.

---

## 2. Codebase Structure
The project is organized into a modular hierarchy:

```text
ai-engine-git-v4/
├── src/
│   ├── model/          # Neural Network Architectures
│   ├── train/          # Training scripts (Stage 1, 2, 3)
│   ├── loss/           # Loss functions (RD, Perceptual, GAN)
│   ├── utils/          # Metrics, EMA, and Entropy Coding
│   └── inference.py    # Testing and Hardware Profiling
├── requirements.txt    # Python dependencies
├── README.md           # Quick start guide
└── models/             # Trained weights (.pth files)
```

---

## 3. Detailed File Analysis

### 📂 `src/model/` (The Brain)
| File | What the code says | Integration / Purpose |
| :--- | :--- | :--- |
| `aether_codec.py` | Defines the `AetherCodec` class, coordinating the encoder, decoder, and hyperprior. | The **entry point** for the model. It handles the forward pass from image to latent and back. |
| `analysis.py` | Implements the **Encoder**. Uses strided convolutions and Swin Transformer blocks to downsample the image into a compact latent representation. | Reduces spatial redundancy and maps RGB pixels to a high-dimensional feature space (latents). |
| `synthesis.py` | Implements the **Decoder**. Features transposed convolutions and a `ResidualRefinementNetwork` (RRN) to reconstruct the image. | Maps the quantized latents back to RGB space, cleaning up artifacts via the RRN. |
| `hyperprior.py` | A "network within a network." It models the probability distribution (entropy) of the latents using a Mean-Scale GMM. | **Predicts the bits needed** to store the image. It uses a side-channel (z-latent) to help the decoder understand the latent distribution. |
| `attention.py` | Implements Swin Transformer blocks with Window-based Multi-Head Self-Attention. | Provides **long-range spatial modeling**, allowing the codec to understand global image structure better than standard CNNs. |
| `quantizer.py` | Implements `SovereignQuantizer` using Straight-Through Estimation (STE). | Converts continuous neural features into **discrete integers** that can be saved to a file. |
| `discriminator.py` | A `MultiScaleDiscriminator` that looks at the image at different resolutions. | Used during Stage 3 training to **force the decoder to generate sharp details** rather than blurry averages. |
| `qvs_flow.py` | Implements a unitary coupling layer for the hyperprior. | Enhances the entropy model's ability to capture complex dependencies in the latent space. |

### 📂 `src/loss/` (The Objectives)
| File | What the code says | Integration / Purpose |
| :--- | :--- | :--- |
| `rate_distortion.py`| The core LIC loss: $L = \lambda \cdot Distortion + Rate$. | Balances the **trade-off between file size (Rate) and quality (Distortion)**. |
| `perceptual.py` | Uses a pre-trained AlexNet (LPIPS) to compare images. | Ensures the reconstruction **feels right to human eyes**, even if pixels aren't an exact match. |
| `adversarial.py` | Implements Generator and Discriminator GAN losses. | Powers the "Elite" stage to produce **high-frequency textures** and realistic details. |

### 📂 `src/train/` (The Education)
| File | What the code says | Integration / Purpose |
| :--- | :--- | :--- |
| `dataset.py` | Custom PyTorch dataset for loading high-res images (e.g., DIV2K). | Feeds the pipeline with training data. |
| `stage1.py` | Basic training: Rate + MSE. | Builds the **foundation** of the codec (compression logic). |
| `stage2.py` | Refinement: Rate + MS-SSIM. | Fine-tunes the model for **better structural similarity** scores. |
| `stage3.py` | Elite Training: Rate + GAN + LPIPS. | The final polish for **production-level perceptual quality**. |

---

## 4. System Integration Workflow
1.  **Encoding**: The `AnalysisTransform` (`analysis.py`) turns a 3-channel image into a 192-channel latent $y$.
2.  **Entropy Modeling**: The `Hyperprior` (`hyperprior.py`) analyzes $y$ to create a probability map.
3.  **Quantization**: The `SovereignQuantizer` (`quantizer.py`) rounds the values.
4.  **Decoding**: The `SynthesisTransform` (`synthesis.py`) takes the rounded values and "hallucinates" the original image back.
5.  **Refinement**: The `ResidualRefinementNetwork` in the synthesis stage performs a final pass to fix edges and textures.

---

## 5. Developer Guide

### Prerequisites
Install the required packages:
```bash
pip install -r requirements.txt
```

### Training
The model is trained progressively to ensure stability:
1.  **Stage 1**: `python -m src.train.stage1 --epochs 40` (Foundation)
2.  **Stage 2**: `python -m src.train.stage2 --epochs 30` (Refinement)
3.  **Stage 3**: `python -m src.train.stage3 --epochs 25` (Elite/GAN)

### Inference & Testing
To test the model on a local image and get a hardware performance report:
```bash
python src/inference.py --checkpoint stage3_elite_final.pth
```
This script will:
- Calculate real BPP (Bits Per Pixel).
- Measure VRAM usage.
- Estimate mobile performance (latency).
- Save a side-by-side comparison.

---

**Documentation generated by Antigravity AI.**
