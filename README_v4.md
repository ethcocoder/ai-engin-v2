# AetherCodec-Elite: AI Engine V4 Mathematical Architecture

AetherCodec-Elite is a state-of-the-art learned image codec that transforms the legacy VAE framework into a mathematically rigorous **Mean-Scale Hyperprior** architecture. It incorporates Swin Transformers for spatial modeling, a repurposed Quantum Virtual Substrate (QVS) for unitary coupling, and a multi-stage training pipeline for elite perceptual quality.

## Mathematical Foundation

### 1. Mean-Scale Hyperprior Framework
The codec is built on the variational autoencoder framework optimized for rate-distortion performance. It uses a hyperprior to model the spatial dependencies in the latent representation $y$.
- **Latent $y$**: $y = f_a(x)$, where $f_a$ is the Analysis Transform.
- **Hyper-latent $z$**: $z = h_a(y)$, where $h_a$ is the Hyper-analysis Transform.
- **Entropy Model**: The distribution of the quantized latent $\hat{y}$ is modeled as a conditional Gaussian Mixture Model (GMM):
  $p(\hat{y} | \hat{z}) = \sum_{i=1}^K w_i \mathcal{N}(\hat{y}; \mu_i, \sigma_i^2)$
  where the parameters $(w_i, \mu_i, \sigma_i)$ are predicted by the Hyper-synthesis network $h_s(\hat{z})$ and a spatial context model.

### 2. QVS Unitary Coupling
The **Quantum Virtual Substrate (QVS)** is repurposed as a learned affine coupling layer within the hyperprior. It implements a unitary (orthogonal) $1 \times 1$ convolution using the **Cayley parametrization**:
$W = (I - A)(I + A)^{-1}$
where $A$ is a skew-symmetric matrix ($A = -A^T$). This transformation preserves the volume of the hyper-latent space, mimicking quantum probability conservation.

### 3. SovereignQuantizer (v4)
Quantization is handled by a learnable per-channel step size $\Delta$:
- **Training**: $\hat{y} = y + \mathcal{U}(-0.5, 0.5) \cdot \Delta$ (Soft quantization via uniform noise).
- **Inference**: $\hat{y} = \text{round}(y / \Delta) \cdot \Delta$ (Hard quantization with Straight-Through Estimator).

## Architecture Components

### Analysis Transform (Encoder)
- 4 stages of downsampling (stride-2 convolutions) with residual blocks.
- **Swin Transformer** blocks at the lowest resolution ($H/16, W/16$) to capture long-range dependencies.
- Output latent $y$ shape: $(B, 192, H/16, W/16)$.

### Synthesis Transform (Decoder)
- Mirror of the encoder with Swin blocks and transpose-convolutions.
- **Residual Refinement Network (RRN)**: A post-processing module that takes $[x_{recon}, x_{orig}]$ and outputs a residual correction map to restore high-frequency details.

### Rate-Distortion Loss
The total loss is defined as:
$L = \lambda \cdot D + R$
- **Rate $R$**: Estimated bits-per-pixel (bpp) from the negative log-likelihoods of $\hat{y}$ and $\hat{z}$.
- **Distortion $D$**: $L_1(x, \hat{x}) + 0.5 \cdot (1 - \text{MS-SSIM}(x, \hat{x})) + 0.1 \cdot \text{LPIPS}(x, \hat{x})$.

## Training Pipeline

The model is trained in three distinct stages:
1. **Stage 1 (Foundation)**: Train encoder, decoder, and hyperprior using $R + \lambda \cdot \text{MSE}$ ($\lambda=0.01$).
2. **Stage 2 (Refinement)**: Freeze hyperprior, switch distortion to MS-SSIM ($\lambda=0.05$).
3. **Stage 3 (Elite)**: Add PatchGAN discriminator and LPIPS loss for maximum perceptual quality ($\lambda=0.1$).

## Evaluation Metrics
- **bpp**: Actual bitrate calculated from the entropy coder.
- **PSNR / MS-SSIM**: Standard distortion metrics.
- **LPIPS**: Perceptual similarity using AlexNet features.
- **FID**: Generative quality metric.

---
*AetherCodec-Elite: Redefining the boundaries of learned image compression.*
