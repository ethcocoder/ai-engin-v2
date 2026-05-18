# AetherCodec-Elite: P2P Transmission Protocol (v7 - Blueprint Edition)
=======================================================================

This documentation guides you through training and deploying **AetherCodec-Elite** and its high-performance hybrid evolution, **Aether-Blueprint**.

> [!IMPORTANT]
> **v7 Blueprint Upgrades:**
> - **TiledHybridCodec**: Seamlessly blends Mathematical Rules with Neural Intelligence.
> - **Rule-Based Mode**: Simple tiles (sky, flat walls) are dynamically encoded using elite 8th-degree Chebyshev polynomials (~540 bytes/tile), completely eliminating edge artifacts to guarantee a flawless 25-28 dB baseline.
> - **Neural Mode**: Complex tiles (faces, texture) use the full Transformer engine (~4-30 KB/tile).
> - **Gaussian Alpha-Blending**: Guaranteed invisible seams between tiles.

---

## ⚡ 1. Setup Environment (Run Once)
```python
# Clone the Elite v7 Branch
!git clone -b v6 https://github.com/ethcocoder/ai-engin-v2.git
%cd ai-engin-v2

# Install Expert Dependencies
!pip install torch torchvision torchmetrics torchac lpips tqdm pillow
```

---

## 🌌 2. Training the Neural Engine
*The Blueprint hybrid system uses a pre-trained AetherCodec as its "Complex Specialist".*

### Stage 1: Foundation (λ=0.01)
```python
!python src/train/stage1.py --epochs 10 --batch_size 16 --lmbda 0.01 --data_dir auto
```

### Stage 2 & 3: Refinement (MS-SSIM/GAN)
```python
!python src/train/stage2.py --epochs 25 --batch_size 16 --data_dir auto
!python src/train/stage3.py --epochs 50 --batch_size 8 --data_dir auto
```

---

## 🔬 3. Evaluating the "Blueprint" Hybrid
*Use the new hybrid inference to achieve extreme compression ratios.*

### Running Hybrid Inference
```python
# Standard Hybrid (Balanced)
!python src/inference_hybrid.py --image test_hd.jpg --model stage1_foundation.pth --threshold 50.0

# 🎲 Test on a Random Image from your Dataset
!python src/inference_hybrid.py --image random --model stage1_foundation.pth --threshold 50.0

# Aggressive Hybrid (More Math, Smaller Files)
!python src/inference_hybrid.py --image test_hd.jpg --model stage1_foundation.pth --threshold 20.0

# Extreme Quality (More Neural, Larger Files)
!python src/inference_hybrid.py --image test_hd.jpg --model stage1_foundation.pth --threshold 150.0
```

### Understanding the Results
| Metric | AetherCodec (Neural Only) | Aether-Blueprint (Hybrid) |
|--------|--------------------------|---------------------------|
| **Format** | .padox | .bpox |
| **Simple Areas** | Neural (Expensive) | Chebyshev Math (~540 bytes) |
| **Complex Areas** | Neural | Neural |
| **Stitching** | N/A (Full Image) | Gaussian Overlap Blend |
| **Ratio** | 15-50x | 30-150x |

---

## 🛡️ Best Practices for Hybrid Deployment

1. **Threshold Tuning**: A threshold of **50.0** is the "Golden Ratio". If the image looks blurry in smooth areas, decrease the threshold. If the file is too large, increase it.
2. **512x512 Native**: The Blueprint is optimized for 512x512 tiles. Images will be automatically resized or tiled to fit this grid.
3. **P2P Streaming**: Use the `.bpox` format for chat apps to minimize bandwidth while maintaining "Elite" visual quality.

---

## 🛠️ Troubleshooting

- **Visible Seams**: Ensure the `overlap` parameter matches in both `TiledHybridCodec` and `TileManager` (default: 16px).
- **NaN Loss**: The Cayley Transform in `qvs_flow.py` is numerically stable, but ensure your learning rate is not above `1e-4`.
- **Hybrid Performance**: Math rendering is CPU-efficient; Neural decoding remains the primary GPU bottleneck.
