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

### Step 1: Download a Training Dataset
We will use an image dataset (like DIV2K or COCO) to teach the Engine.
```bash
# Download a sample dataset of images to train on
!mkdir dataset
!wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
!unzip -q DIV2K_train_HR.zip -d dataset/
```

### Step 2: Train the Core Mathematics (Stage 1 & 2)
This teaches the Sender how to compress into 4-16KB payloads.
```python
# Stage 1: Build the Latent Math
!python -c "from src.train.stage1 import train_stage1; from src.model.aether_codec import AetherCodec; from src.train.dataset import get_dataloader; model = AetherCodec(); loader = get_dataloader('dataset/DIV2K_train_HR', batch_size=8); train_stage1(model, loader, epochs=100)"

# Stage 2: Refine Structural Perception
!python -c "from src.train.stage2 import train_stage2; from src.model.aether_codec import AetherCodec; from src.train.dataset import get_dataloader; model = AetherCodec(); loader = get_dataloader('dataset/DIV2K_train_HR', batch_size=8); train_stage2(model, loader, epochs=100)"
```

### Step 3: Train the Receiver GAN (Stage 3)
This is where the magic happens. The GAN learns to take the compressed math and synthesize HD images.
```python
!python -c "from src.train.stage3 import train_stage3; from src.model.aether_codec import AetherCodec; from src.train.dataset import get_dataloader; model = AetherCodec(); loader = get_dataloader('dataset/DIV2K_train_HR', batch_size=8); train_stage3(model, loader, epochs=50)"
```

---

## 🛡️ Best Practices for Telegram-Style Deployment
Once trained, you detach the Encoder and the Decoder:
- **Mobile App (Sender):** Only runs the `AnalysisTransform` (Encoder).
- **Mobile App (Receiver):** Only runs the `SynthesisTransform` (Decoder GAN).
- **The Protocol:** Use `torchac` (Arithmetic Coding) to compress the output of the Sender into the final 16KB `.pdox` file sent over the chat network.
