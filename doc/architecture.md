# Neural Architectural Blueprint

## Overview
The Telecom AI Engine replaces traditional block-compression formats (JPEG/WebP) with a deep **Spatial Neural Compressor**. Instead of compressing raw pixels mathematically using Discrete Cosine Transforms (DCT), we utilize highly parameterized Convolutional Neural Networks (CNNs) to extract the *meaning* and *geometry* of the image into a highly dense abstract bottleneck.

## Core Components (`model.py`)

### 1. SpatialEncoder (Mobile Device A)
* **Goal**: Absorb an HD Tensor (`3 x H x W`) and crush it down into a highly compressed transmission payload (`C x (H/8) x (W/8)`).
* **Architecture**: Features a series of cascading `Conv2d` layers with `stride=2` to downsample the mathematical footprint. Between downsamples, it executes a `ResBlock` (Residual skip-connection) to guarantee that absolute edge detection and boundary features are never lost into the gradient.
* **Output**: A 2D spatial coordinate map. We maintain 2D arrays heavily instead of flattening it to 1D, meaning X/Y axes geometry is cleanly preserved for decoding.

### 2. SpatialDecoder (Mobile Device B)
* **Goal**: Receive the incoming raw transmission payload (from the internet) and hallucinate the HD resolution symmetrically.
* **Architecture**: Uses `ConvTranspose2d` (Deconvolution) arrays to symmetrically rebuild the spatial blocks. Residual blocks inject high-frequency fidelity bounds back into the image. Output bounded by a `Tanh` threshold mapped back to visual spectrum arrays.

## Perceptual Error Calculation (`train.py`)
Standard autoencoders suffer from generic image "blurriness." This is due to pure MSE (Mean Squared Error) usage.
To combat this, the architecture fuses **L1 Loss** (Sharpness) with **MSE Loss** to mathematically penalize the AI if it fails to recreate microscopic contrast correctly.
