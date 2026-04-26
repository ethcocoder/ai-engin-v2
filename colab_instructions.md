# AetherCodec-Elite: Telegram-Style Transmission Protocol (v4)
=====================================================

This repository implements **AetherCodec-Elite**, designed specifically for highly constrained P2P / chat environments (like Telegram). 

## 🧠 The Concept: Mathematics-to-Image Synthesis
Your idea of compressing an image into "mathematics components" is exactly what this engine does! 
1. **The Sender (Encoder):** Runs locally on a device. It takes an image and converts it into a quantized latent tensor (the "mathematics"). 
2. **The Payload (Entropy Coder):** This math is compressed into a binary file roughly **4KB to 16KB** in size.
3. **The Receiver (GAN Decoder):** Accepts the 4KB-16KB math and uses an Adversarial GAN to perfectly hallucinate and synthesize the original image, recovering textures and edges that would normally require megabytes of data.

---

## ⚡ Hardware Setup
- **GPU**: NVIDIA T4 (Default in Colab)
- **TPU**: Google TPU v5e (supported via PyTorch XLA)

```bash
# 1. Clone the AetherCodec-Elite Repository
!git clone -b v4 https://github.com/ethcocoder/ai-engin-v2.git
%cd ai-engin-v2

# 2. Install Dependencies
!pip install -r requirements.txt
!pip install torch torchvision torchmetrics torchac
```

---

## 🌌 The Training Pipeline

Because we need the Receiver to learn how to hallucinate from 16KB of math, we must train the system. 
*Note: We have added a built-in dataloader so you no longer need to write your own!*

### Step 1: Train the Core Mathematics (Stage 1)
The dataset **auto-downloads** on first run. Now optimized for **40 epochs**.
```bash
# Optimized run (40 epochs default)
!python src/train/stage1.py
```
💾 **Checkpoint saved to:** `stage1_foundation.pth`

### Step 2: Refine Structural Perception (Stage 2)
Automatically loads `stage1_foundation.pth`. Optimized for **30 epochs**.
```bash
# Optimized run (30 epochs default)
!python src/train/stage2.py
```
💾 **Checkpoint saved to:** `stage2_refined.pth`

### Step 3: Train the Receiver GAN (Stage 3)
The final photorealistic stage. Optimized for **25 epochs**.
```bash
# Optimized run (25 epochs default)
!python src/train/stage3.py
```
💾 **Checkpoint saved to:** `stage3_elite_final.pth`

### Step 4: Progression Study (The "Aether Evolution")
You can now test the model after each stage to see the quality grow. Each command will save a specific result image (e.g., `result_stage1_foundation.png`).

```bash
# Test Stage 1 Foundation
!python src/inference.py --checkpoint stage1_foundation.pth

# Test Stage 2 Refinement
!python src/inference.py --checkpoint stage2_refined.pth

# Test Stage 3 Final (The "Elite" GAN version)
!python src/inference.py --checkpoint stage3_elite_final.pth
```


---

## 🛡️ Best Practices for Telegram-Style Deployment
Once trained, you detach the Encoder and the Decoder:
- **Mobile App (Sender):** Only runs the `AnalysisTransform` (Encoder).
- **Mobile App (Receiver):** Only runs the `SynthesisTransform` (Decoder GAN).
- **The Protocol:** Use `torchac` (Arithmetic Coding) to compress the output of the Sender into the final 16KB `.pdox` file sent over the chat network.
