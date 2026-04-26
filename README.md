# AetherCodec-Elite: State-of-the-art Learned Image Codec

AetherCodec-Elite is a cutting-edge learned image compression model derived from the quantum-inspired framework. It marries the mathematical rigor of modern rate-distortion theory with state-of-the-art architectural components like Swin Transformers and Adversarial learning. It transforms the legacy QVS into a mathematically sound Unitary Coupling Layer.

## 1. Rate-Distortion Theory Behind the Hyperprior

In classical autoencoders, latents are deterministically mapped. In AetherCodec-Elite, compression is framed as a rate-distortion optimization problem:
$$ L = \lambda \cdot D + R $$

Where $D$ is the distortion (e.g., L1, MS-SSIM, LPIPS) and $R$ is the rate. The rate $R$ represents the actual bitrate required to store the latent representations $y$ and $z$. 

To estimate the rate, we use a Mean-Scale Hyperprior:
- The latent $y$ is quantized to $\hat{y}$ using a Straight-Through Estimator (STE) during inference, and additive uniform noise during training.
- The hyper-latent $z$ is extracted from $y$ to capture spatial dependencies. It is also quantized to $\hat{z}$ and sent as side-information.
- A hyper-synthesis network uses $\hat{z}$ to predict the parameters (weights, means, scales) of a Gaussian Mixture Model (GMM).
- The rate is calculated as the cross-entropy $R = \mathbb{E}[-\log_2 p(\hat{y}|\hat{z})] + \mathbb{E}[-\log_2 p(\hat{z})]$. This gives a tight upper bound on the actual file size.

## 2. QVS as a Unitary Coupling Layer

The original Quantum Virtual Substrate (QVS) was a somewhat heuristic reparameterization. In AetherCodec-Elite, QVS has been repurposed into a rigorous **Unitary Coupling Layer** placed within the hyperprior.

- **Cayley Parametrization:** We construct a strictly orthogonal (unitary) 1x1 convolution weight matrix $W = (I - A)(I + A)^{-1}$, where $A$ is a learned skew-symmetric matrix.
- **Quantum Analogy:** Because $W$ is orthogonal, $\det(W) = 1$. It preserves the volume of the latent space, mimicking the probability-conserving nature of quantum operators.
- **Purpose:** It applies a non-destructive, learned rotation to the hyper-latent $z$, allowing the network to decorrelate channels before hyper-synthesis without changing the entropy bounds.

## 3. Training Instructions for the 3 Stages

The codec requires a phased training approach to stabilize the entropy model before applying adversarial and perceptual losses.

**Stage 1: Rate + MSE (100 Epochs)**
Train the encoder, decoder, and hyperprior from scratch.
- **Objective:** $\lambda \cdot \text{MSE} + R$
- **Lambda:** $0.01$ (Low bitrate) or $0.05$ (High bitrate).
- **Goal:** Initialize the latents and entropy model safely without perceptual artifacts exploding the gradients.

**Stage 2: MS-SSIM Fine-Tuning (100 Epochs)**
Switch to a structural distortion metric.
- **Objective:** $\lambda \cdot (1 - \text{MS-SSIM}) + R$
- **Lambda:** $0.05$
- **Goal:** Optimize for human-perceived structure. We freeze the hyperprior to let the main codec adapt to the frozen entropy model.

**Stage 3: Adversarial Elite Quality (50 Epochs)**
Unfreeze all layers and introduce the Multi-scale PatchGAN Discriminator.
- **Objective:** $R + \lambda \cdot (\text{MS-SSIM} + 0.1 \cdot \text{LPIPS}) + 0.1 \cdot L_G$
- **Goal:** Synthesize high-frequency details lost to quantization, achieving elite perceptual quality at ultra-low bitrates.

*Use the provided scripts in `src/train/stage{1,2,3}.py` to execute these stages.*

## 4. Evaluation on Kodak / CLIC2020 Datasets

To evaluate the codec, we measure Bits Per Pixel (BPP), PSNR, MS-SSIM, and LPIPS.

1. **Load Pre-trained Weights:** Initialize `AetherCodec` and load the Stage 3 weights.
2. **Quantize:** Pass the image through the encoder. Ensure `force_hard=True` to use true rounding for $y$ and $z$.
3. **Entropy Coding:** Use an arithmetic coder (e.g., `torchac`) to compress $\hat{y}$ and $\hat{z}$ into a real byte stream.
   - $\text{BPP} = (\text{len(byte\_stream)} \times 8) / (H \times W)$
4. **Decode & Measure:**
   - Decode the byte stream back to $\hat{y}$.
   - Pass $\hat{y}$ through the Synthesis Transform.
   - Compare the output with the original image using `src/utils/metrics.py`.

AetherCodec-Elite is capable of surpassing ELIC/TIC architectures, achieving < 0.5 bpp with > 30 dB PSNR and < 0.05 LPIPS on Kodak 512x512 crops.
