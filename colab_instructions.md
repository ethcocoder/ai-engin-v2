# AetherCodec-Elite: P2P Transmission Protocol (v6 - Compression Fix)
====================================================================

This documentation guides you through training and deploying the **Honest AetherCodec-Elite**, a mathematically rigorous neural image transmission system optimized for high-fidelity chat communication.

> [!IMPORTANT]
> **v6 Critical Fixes Applied:**
> - Entropy coder now properly quantizes latents to integers before serialization (was storing raw floats as int16)
> - Quantizer step size increased from 0.1→1.0 (reduces entropy by ~10x = much smaller .padox files)
> - Rate warmup shortened from 30%→10% of epochs (compression pressure kicks in immediately)
> - Inference pipeline now reports compression ratio and verifies latent reconstruction fidelity

---

## ⚡ 1. Setup Environment (Run Once)
### Option A: Clone from GitHub (Recommended)
```python
# Clone the Elite v6 Branch
!git clone -b v6 https://github.com/ethcocoder/ai-engin-v2.git
%cd ai-engin-v2

# Install Expert Dependencies
!pip install torch torchvision torchmetrics torchac lpips tqdm pillow
!pip install -U --no-cache-dir gdown

# Verify CUDA (Crucial for Swin Transformers)
import torch
print(f"✅ PyTorch {torch.__version__} initialized on {torch.cuda.get_device_name(0)}")
```

### Option B: Manual Upload
1. Zip your local `src` folder and upload it to Colab.
2. Run: `!unzip your_code.zip`

---

## 🌌 2. The 3-Stage Training Pipeline
*Run these stages sequentially. Each stage builds upon the previous one.*

### Understanding Lambda (λ) — The Compression Control Knob

| Lambda | Mode | Expected .padox Size (512×512) | Expected PSNR |
|--------|------|-------------------------------|---------------|
| 0.001  | Extreme compression | 2-8 KB | 24-27 dB |
| 0.01   | Aggressive compression | 8-30 KB | 27-30 dB |
| 0.05   | Balanced quality/size | 30-80 KB | 30-34 dB |
| 0.1    | High quality | 80-200 KB | 34-38 dB |

> [!TIP]
> For your goal of **2MB image → 4KB .padox**, use `--lmbda 0.001` (extreme compression).
> For usable chat quality, use `--lmbda 0.01` (aggressive but recognizable).

### Stage 1: The Foundation (Core Compression)
*Focus: Learning the base latent representation and GMM entropy model.*

> [!IMPORTANT]
> **Minimum 50 epochs recommended.** 10 epochs is far too few for the model to learn compression.
> Using `--data_dir auto` will download ~8.5GB of data (DIV2K, CLIC, COCO, Flickr).

```python
# AGGRESSIVE COMPRESSION (small files, lower quality):
!python src/train/stage1.py --epochs 50 --batch_size 16 --lmbda 0.01 --data_dir auto

# EXTREME COMPRESSION (tiny files, visible artifacts):
# !python src/train/stage1.py --epochs 50 --batch_size 16 --lmbda 0.001 --data_dir auto

# BALANCED (good quality, moderate files):
# !python src/train/stage1.py --epochs 50 --batch_size 16 --lmbda 0.05 --data_dir auto
```

**What to watch during training:**
- `bpp` should decrease over epochs (model learning to compress)
- `dist` should stay reasonable (not exploding)
- `step` shows average quantization step (higher = more compression)

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
*Verify each stage to ensure the compression pipeline is working correctly.*

```python
# 0. Download a High-Definition sample image for testing
!wget -O test_hd.jpg "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=2074&auto=format&fit=crop"

# 1. Verify Stage 1: Base Compression (Check .padox file size!)
!python src/inference.py --image test_hd.jpg --model stage1_foundation.pth

# 2. Verify Stage 2: Structural Refinement (SSIM/LPIPS should improve)
!python src/inference.py --image test_hd.jpg --model stage2_refined.pth

# 3. Verify Stage 3: Full "Elite" Photorealism
!python src/inference.py --image test_hd.jpg --model stage3_elite_final.pth
```

**Expected output after Stage 1 with λ=0.01:**
```
📊 ELITE INFERENCE RESULTS
==================================================
  Source        : test_hd.jpg
  Original Size : 400.00 KB
  .padox Size   : 15.23 KB        ← Should be MUCH smaller than original
  Compression   : 26.3x (96.2% reduction)
  PSNR          : 28.50 dB        ← Decent quality
  MS-SSIM       : 0.9200
  Recon Error   : 0.000000        ← Must be 0 (entropy coder integrity)
==================================================
```

> [!WARNING]
> If `Recon Error` is NOT 0.000000, the entropy coder has a bug.
> If `.padox Size` is LARGER than Original Size, the model needs more training epochs.

---

## 🛡️ Expected Quality Targets

For a 512x512 image with λ=0.01:

*   **File Size:** 10-30 KB (.padox)
*   **Compression Ratio:** 15-50x
*   **PSNR:** 27-31 dB
*   **MS-SSIM:** 0.90 - 0.96
*   **LPIPS:** 0.02 - 0.06
*   **P2P Bandwidth:** ~30-50% additional reduction via Delta Coding

### 📱 Deployment Notes:
*   **Sender (App A):** Only needs the `encoder`, `DSQ Quantizer`, and `hyperprior`.
*   **Receiver (App B):** Only needs the `decoder` (SynthesisTransform). Reconstructs solely from the received bitstream.
*   **Protocol:** Uses `ParadoxEntropyCoder` with `zlib` (production: replace with `torchac` for ~20% better compression).
*   **P2P Sync:** `SharedHyperpriorState` auto-detects drift and refreshes keyframes every 30 frames.

---

## 🛠️ Troubleshooting

### File too large?
- Increase training epochs (50→100+)
- Decrease lambda (0.01→0.005→0.001)
- Check that `step` in training log is increasing (should reach 1.0-4.0)

### Quality too low?
- Increase lambda (0.01→0.05→0.1)
- Train more epochs
- Progress to Stage 2 and Stage 3

### Numerical Stability
*   The v6 build uses **Cayley Transform (linalg.solve)** and **Softplus** activations to prevent NaNs.
*   QVS Orthogonality warnings are normal — the model monitors unitary consistency automatically.

### OOM (Out of Memory)
*   Decrease `batch_size` to 4 or 8 in Stage 3.
*   The RRN hidden dimension is tuned to 64 to save VRAM.

### Desync in P2P mode
*   Ensure `DeltaLatentCodec.reset_session()` is called between different image sequences.
