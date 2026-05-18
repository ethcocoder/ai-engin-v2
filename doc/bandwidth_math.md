# Information Theory & The Cost Dial

## Shannon's Source Coding Theorem
When operating this network, you govern the apex intersection between **Server Bandwidth Profit** and **Visual Quality**. In Information Theory, there is a physical limit to how tightly data can be squeezed before irreversible degradation occurs.

## The `latent_channels` Control 
The primary dial you manipulate in this architecture is the `--latent_channels` argument. 

Because the `SpatialEncoder` downsamples spatial scale by `8x` (using 3 stride-2 layers), the final spatial dimensions are `H/8 x W/8`. The `latent_channels` define the depth of that tiny map.

### 1. High-Profit Margin Mode (`latent_channels = 4`)
* **Behavior:** Chokes the information funnel down drastically.
* **Bandwidth Usage:** Extremely light. (e.g. `12.0x Compression Reduction`)
* **Visual Output:** Identical spatial layout, but micro-textures (like grass blades or individual bricks) may exhibit softness. Perfect for low-data limit environments.

### 2. Lossless Luxury Mode (`latent_channels = 32`)
* **Behavior:** Widens the payload vector significantly to permit deep frequency retention.
* **Bandwidth Usage:** Much heavier, yet still profitable compared to original (e.g. `1.5x - 3x Reduction`).
* **Visual Output:** Near-mathematical reconstruction perfection. High frequency textures survive the trip verbatim. Use this setting when User Experience demands flawless pixel preservation.
