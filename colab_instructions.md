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
This teaches the Sender how to compress into 4-16KB payloads. The dataset will **automatically download** the first time you run this!
```bash
# Train Stage 1 (Outputs stage1_foundation.pth)
!python src/train/stage1.py
```

### Step 2: Refine Structural Perception (Stage 2)
Next, we freeze the entropy model and focus on MS-SSIM quality. It will automatically load the weights from Stage 1.
```bash
# Train Stage 2 (Outputs stage2_refined.pth)
!python src/train/stage2.py
```

### Step 3: Train the Receiver GAN (Stage 3)
This is where the magic happens. The GAN learns to take the compressed math and synthesize HD images. It will automatically load the weights from Stage 2.
```bash
# Train Stage 3 (Outputs stage3_elite_final.pth)
!python src/train/stage3.py
```

### Step 4: Test the Telegram-Style Pipeline (Inference)
Now that training is done and the checkpoint is saved (`stage3_elite_final.pth`), let's test the complete pipeline. We will grab an image, run it through the Sender, calculate the math payload size, and run it through the Receiver.
```bash
# Tests Sender -> Transmission -> Receiver Pipeline
!python src/inference.py
```

---

## 🛡️ Best Practices for Telegram-Style Deployment
Once trained, you detach the Encoder and the Decoder:
- **Mobile App (Sender):** Only runs the `AnalysisTransform` (Encoder).
- **Mobile App (Receiver):** Only runs the `SynthesisTransform` (Decoder GAN).
- **The Protocol:** Use `torchac` (Arithmetic Coding) to compress the output of the Sender into the final 16KB `.pdox` file sent over the chat network.
