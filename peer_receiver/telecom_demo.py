"""
telecom_demo.py — Paradox Aether Mesh Bandwidth Simulator
==========================================================
Demonstrates the Genesis Core's neural image compression pipeline as it
would function in a real-world mobile telecom context:

    [User A / Sender]   → encodes image to quantised latent payload
    [Telecom Gateway]   → transmits the compressed neural payload
    [User B / Receiver] → decodes payload back to a full image locally

Produces `telecom_simulation_result.png` comparing originals vs reconstructions
with PSNR quality scores per image.

Usage:
    python src/telecom_demo.py --model_path checkpoints/best_genesis_core.pth
                               --latent_channels 8
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Colab / headless servers
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── Advanced Pathing Protocol ────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model import LatentGenesisCore  # noqa: E402 (must follow path setup)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_NUM_SAMPLES = 4        # Images to compress and visualise
_OUTPUT_FILE = "telecom_simulation_result.png"
_DPI         = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def unnorm(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalise an image tensor from [-1, 1] → [0, 1] for display.

    Args:
        img: Tensor with values in [-1, 1].

    Returns:
        Clamped tensor with values in [0, 1].
    """
    return torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Both tensors must be in the [0, 1] range.
    Higher is better: >25 dB = acceptable, >28 dB = good, >32 dB = excellent.

    Args:
        original:      Ground-truth image tensor.
        reconstructed: Decoded/reconstructed image tensor.

    Returns:
        PSNR in decibels (float). Returns +inf if the images are identical.
    """
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return -10.0 * (torch.log10(torch.tensor(mse))).item()


# ── Simulation ────────────────────────────────────────────────────────────────

def run_bandwidth_simulation(args: argparse.Namespace) -> None:
    """
    Execute the full neural compression simulation pipeline.

    Args:
        args: Parsed CLI arguments (model_path, latent_channels).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[*] Initialising Telecom Neural Compression Simulator on %s\n", device)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model = LatentGenesisCore(latent_channels=args.latent_channels).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info(
        "[*] Loaded checkpoint from epoch %d (best loss: %.4f)",
        checkpoint.get("epoch", "?"),
        checkpoint.get("best_loss", float("nan")),
    )

    # ── 2. Load sample data ───────────────────────────────────────────────────
    # NOTE: Do NOT resize images here. The model was trained on native 32×32
    # CIFAR-10 images. Upscaling at inference time causes catastrophic
    # distribution shift and produces garbage reconstructions.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    loader  = DataLoader(dataset, batch_size=_NUM_SAMPLES, shuffle=True)
    images, _ = next(iter(loader))
    images = images.to(device)

    # ── 3. Sender: encode ─────────────────────────────────────────────────────
    original_bytes = images[0].element_size() * images[0].nelement()
    log.info("[USER A: SENDER MOBILE DEVICE]")
    log.info("[-] Image dimensions (native): %s", list(images[0].shape))
    log.info("[-] Uncompressed payload size: %s bytes", f"{original_bytes:,}")

    encoded_latents: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(_NUM_SAMPLES):
            mu, _ = model.encoder(images[i].unsqueeze(0))
            # Clamp before 8-bit quantisation (matches model.py's STE)
            mu_clamped = torch.clamp(mu, -1.0, 1.0)
            q_latent   = torch.round(mu_clamped * 127.5) / 127.5
            encoded_latents.append(q_latent)

    # ── 4. Gateway: transmission stats ───────────────────────────────────────
    payload_bytes     = encoded_latents[0].nelement() * 1  # 8-bit = 1 byte/element
    compression_ratio = original_bytes / payload_bytes
    latent_shape      = list(encoded_latents[0].squeeze(0).shape)

    log.info("\n[TELECOM GATEWAY: TRANSMISSION]")
    log.info("[-] Neural payload dimensions:         %s", latent_shape)
    log.info("[-] Compressed payload size:           %s bytes", f"{payload_bytes:,}")
    log.info("[-] *** BANDWIDTH REDUCTION:           %.1fx ***", compression_ratio)

    # ── 5. Receiver: decode ───────────────────────────────────────────────────
    log.info("\n[USER B: RECEIVER MOBILE DEVICE]")
    log.info("[-] Collapsing & decoding payload locally...")

    decoded_images: list[torch.Tensor] = []
    psnr_scores:    list[float]        = []
    with torch.no_grad():
        for i in range(_NUM_SAMPLES):
            recon = model.decoder(encoded_latents[i]).cpu().squeeze(0)
            decoded_images.append(recon)
            psnr_scores.append(
                compute_psnr(unnorm(images[i].cpu()), unnorm(recon))
            )

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    log.info("[-] Average PSNR: %.2f dB  (>25 dB = good quality)", avg_psnr)

    # ── 6. Visualisation ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, _NUM_SAMPLES, figsize=(4 * _NUM_SAMPLES, 8))
    fig.suptitle(
        f"Future Telecom Network  |  {compression_ratio:.1f}× Compression  |  "
        f"Avg PSNR: {avg_psnr:.2f} dB",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(_NUM_SAMPLES):
        orig_img  = unnorm(images[i].cpu()).permute(1, 2, 0).numpy()
        recon_img = unnorm(decoded_images[i]).permute(1, 2, 0).numpy()

        # Use nearest-neighbour interpolation: crisp pixels, no bilinear blur
        axes[0, i].imshow(orig_img, interpolation="nearest")
        axes[0, i].set_title(f"Sender Original\n[{original_bytes:,} bytes]", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_img, interpolation="nearest")
        axes[1, i].set_title(
            f"Receiver Final\n({payload_bytes:,} bytes | PSNR: {psnr_scores[i]:.1f} dB)",
            fontsize=9,
        )
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(_OUTPUT_FILE, dpi=_DPI, bbox_inches="tight")
    log.info("\n[*] SUCCESS: Result saved to '%s'", _OUTPUT_FILE)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paradox Aether Mesh — Telecom Bandwidth Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to best_genesis_core.pth checkpoint",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=8,
        help="Must match the --latent_channels used during training",
    )
    args = parser.parse_args()
    run_bandwidth_simulation(args)
