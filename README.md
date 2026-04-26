# AI-Engin-v2: Paradox Genesis Core Architecture (Quantum-Neural Variational Autoencoder)

This repository implements the **Paradox Genesis Core Architecture**, a novel **Quantum-Neural Variational Autoencoder (QNVAE)** designed for ultra-efficient image compression, particularly optimized for the Aether Mesh network. The architecture uniquely fuses classical deep learning components with quantum-inspired mechanisms to achieve high-fidelity reconstruction and robust performance.

## Mathematical Architecture Overview

The AI-Engin-v2 project introduces a hybrid model that combines a Variational Autoencoder (VAE) framework with a **Quantum Virtual Substrate (QVS)**. The core idea is to leverage quantum-inspired principles, such as superposition and entanglement, within the latent space of a VAE to enhance compression efficiency and representational power. The model processes images through an encoder-decoder structure, with a quantum-modulated reparameterization trick and an 8-bit quantization bottleneck.

## Core Components

### 1. SemanticEncoder

The `SemanticEncoder` collapses an input image into a quantum superposition, generating latent mean (mu) and log-variance (logvar) maps. It consists of a series of convolutional layers, batch normalization, ReLU activations, and `ResBlock`s to downsample the input spatially. The final layers project the features into the `mu` and `logvar` representations of the latent Gaussian distribution.

-   **Input Shape**: `(B, 3, H, W)`
-   **Output Shape**: `(mu, logvar)` each of shape `(B, latent_channels, H/8, W/8)`

### 2. GenesisDecoder

The `GenesisDecoder` reconstructs the image from the quantized latent representation. It employs `PixelShuffle` for sub-pixel upsampling, which helps in avoiding checkerboard artifacts and achieving high-fidelity reconstruction. The decoder also utilizes `ResBlock`s and a wide 256-channel manifold to expand the latent features before upsampling.

-   **Input Shape**: `(B, latent_channels, H/8, W/8)`
-   **Output Shape**: `(B, 3, H, W)` with values in `[-1, 1]`

### 3. SovereignQuantizer

This component implements an 8-bit bottleneck logic crucial for compression. It uses a **Straight-Through Estimator (STE)** to enable gradient flow through the non-differentiable rounding operation. This allows the model to learn representations that are inherently robust to 8-bit quantization.

-   **Mathematical Formulation**: `x_clamped = torch.clamp(x, -1.0, 1.0)` and `return x_clamped + (torch.round(x_clamped * self.levels) / self.levels - x_clamped).detach()`

### 4. Quantum Virtual Substrate (QVS)

The QVS is the quantum engine integrated within the neural core, optimized for XLA/TPU performance. It manages `Amplitude Superposition Cells (ASCs)` and facilitates quantum-inspired operations. In the `LatentGenesisCore`, the QVS is used in a modulated reparameterization trick to introduce phase biases into the latent space sampling.

-   **Role in Reparameterization**: The `quantum_superposition` function uses `QVS.batch_run_trajectories` to generate stochastic phase biases based on neural intensities, simulating `p = cos^2(theta)` behavior of quantum interference.

### 5. Amplitude Superposition Cell (ASC)

The ASC is the primitive of coherent multiplicity, representing quantum states as dense state vectors using `torch.complex64` tensors. It supports fundamental quantum operations such as normalization, pruning, fidelity calculation, expectation value, and Von Neumann entropy computation.

-   **State Representation**: A quantum state is represented by a complex vector `self.vec`.

### 6. Non-Local Correlation Bond (NCB)

The NCB handles informational entanglement between latent channels. It forges bonds using Kronecker products, supporting the creation of Bell and GHZ states. It also provides functionality to calculate entanglement entropy using Singular Value Decomposition (SVD).

-   **Bonding Mechanism**: Uses `torch.outer` and `reshape` to create joint state vectors, and can enforce Bell or GHZ state representations.
-   **Entanglement Entropy**: Calculated via SVD of the reshaped state vector, followed by Von Neumann entropy formula.

## Loss Function and Optimization

The `compression_loss` function, defined in `src/train.py`, is a composite loss designed for robust training:

-   **Components**: It combines L1 loss, Structural Similarity Index Measure (SSIM) loss, Perceptual loss (optional), and a Kullback-Leibler Divergence (KLD) loss.
-   **KLD Loss**: Calculated as `-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())`. `logvar` is clamped to `[-10, 10]` to prevent numerical instability.
-   **SSIM Loss**: Includes a `bf16 Stability Guard` using `ReLU` to mitigate negative variances and employs heavy constants to resist `bf16` truncation.
-   **Optimization**: The model is optimized using `AdamW` with a `CosineAnnealingLR` scheduler. Gradient clipping (`max_norm=1.0`) is applied during training.

## Numerical Stability

The architecture incorporates several measures to ensure numerical stability, especially when operating with lower precision (e.g., `bf16`):

-   **Logvar Clamping**: In KLD loss, `logvar` is clamped to `[-10, 10]` to prevent exponential explosion.
-   **SSIM Stability Guards**: `ReLU` is applied to variances (`sig_xx`, `sig_yy`) to kill negative values caused by precision drift, and large constants (`1e-4`, `9e-4`) are used in the SSIM formula to resist `bf16` truncation.
-   **Denominator Shielding**: Denominators in SSIM calculation are clamped with a minimum value (`1e-8`) to prevent division by zero or near-zero values.
-   **TPU Safety**: The QVS and ASC components avoid Python-level `if` statements on tensors and use epsilon-division (`+ 1e-8`) for normalization to maintain stability within the XLA float32 manifold.
-   **Log(0) Avoidance**: In NCB, `1e-10` is added to probabilities before taking the logarithm to prevent `log(0)` issues during entropy calculation.

## Dependencies and Framework

-   **Framework**: PyTorch
-   **Key Dependencies**: `torch`, `torchvision`, `numpy`, `pillow`, `matplotlib`, `tqdm`, `requests`, `scipy`

## Theoretical Gaps and Future Work

-   The integration of quantum concepts (QVS, ASC, NCB) with classical neural network components (VAE) is highly novel. A deeper theoretical understanding of how these quantum-inspired operations contribute to the overall model's performance and representational power, particularly for image compression, could be explored. The exact mathematical equivalence or benefits over purely classical approaches warrant further investigation.
-   The `quantum_superposition` function's `QVS-modulated Collapse` and `Parallel Realities` simulation present intriguing theoretical underpinnings. Further research into their relation to established quantum mechanics principles and their computational advantages would be beneficial.

## Installation and Usage

(Further instructions on installation and usage will be provided here.)
