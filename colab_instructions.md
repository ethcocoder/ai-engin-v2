# AetherCodec-Elite: P2P Transmission Protocol (v5 - Master)
====================================================================

This documentation guides you through training and deploying the **Honest AetherCodec-Elite**, a mathematically rigorous neural image transmission system optimized for high-fidelity chat communication.

---

## ⚡ 1. Setup Environment (Run Once)
Copy and paste this block into a Colab cell to initialize the Elite environment.

```python
# 1. Install Expert Dependencies
!pip install torch torchvision torchmetrics torchac lpips tqdm pillow
!pip install -U --no-cache-dir gdown

# 2. Verify CUDA (Crucial for Swin Transformers)
import torch
print(f"✅ PyTorch {torch.__version__} initialized on {torch.cuda.get_device_name(0)}")
```

---

## 🌌 2. The 3-Stage Training Pipeline
*Run these stages sequentially. Each stage builds upon the previous one.*

### Stage 1: The Foundation (Core Compression)
*Focus: Learning the base latent representation and GMM entropy model.*
```python
!python src/train/stage1.py --epochs 25 --batch_size 16 --data_dir auto
```

### Stage 2: Structural Refinement (The Perceptual Eye)
*Focus: Activating LPIPS and MS-SSIM to sharpen structural details.*
```python
!python src/train/stage2.py --epochs 25 --batch_size 16 --data_dir auto
```

### Stage 3: Elite Adversarial Training (Photorealism)
*Focus: Activating the Multi-Scale PatchGAN for high-frequency textures.*
```python
!python src/train/stage3.py --epochs 50 --batch_size 8 --data_dir auto
```

---

## 🔬 3. Real-World P2P Evaluation
*Simulate a real-world transmission: Compress to .padox binary -> Send -> Decode.*

```python
# 1. Test Base Compression
!python src/inference.py --image test.jpg --model stage1_latest.pth

# 2. Test Refined Performance
!python src/inference.py --image test.jpg --model stage2_refined.pth

# 3. Full "Elite" Reconstruction (The Gold Standard)
!python src/inference.py --image test.jpg --model stage3_elite_final.pth
```

---

## 🛡️ Best Practices & "Honesty" Metrics
For a 512x512 image compressed to ~16KB (.padox):

*   **PSNR:** 29-33 dB (Excellent for chat)
*   **MS-SSIM:** 0.94 - 0.98 (Structural perfection)
*   **LPIPS:** 0.015 - 0.04 (Human-perceived quality)

### 📱 Deployment Notes:
*   **Sender (App A):** Only needs the `encoder`, `y_quantizer`, and `hyperprior`. It outputs the `.padox` bitstream.
*   **Receiver (App B):** Only needs the `decoder` (SynthesisTransform). It is strictly "Honest" and reconstructs solely from the received bitstream.
*   **Protocol:** The `ParadoxEntropyCoder` manages the binary conversion, preserving quantization metadata for perfect color reconstruction.

---

## 🛠️ Troubleshooting
*   **NaN Loss:** Ensure you are using the refactored `hyperprior.py` with scale clamping.
*   **OOM (Out of Memory):** Decrease `batch_size` to 4 or 8 in Stage 3.
*   **Blurry Borders:** Verify `src/utils/metrics.py` is using 'same' padding (Elite Audit v5).
