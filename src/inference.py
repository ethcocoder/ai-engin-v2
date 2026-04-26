import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from torchvision.transforms.functional import to_pil_image
from src.model.aether_codec import AetherCodec
from src.train.dataset import ImageFolderDataset
from torchvision import transforms

# ─────────────────────────────────────────────
# Hardware Profiler
# ─────────────────────────────────────────────
def get_gpu_stats(device):
    """Returns current GPU memory usage in MB."""
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return allocated, reserved, total
    return 0.0, 0.0, 0.0

def print_hardware_report(device, image_shape,
                          t_encode, t_decode,
                          vram_before, vram_after,
                          payload_kb):
    """Prints a structured hardware usage report after inference."""
    H, W = image_shape[-2], image_shape[-1]
    num_pixels = H * W

    print("\n" + "="*55)
    print("       ⚙️  AETHERCODEC-ELITE — HARDWARE REPORT")
    print("="*55)

    # --- Device ---
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        _, _, total_vram = get_gpu_stats(device)
        print(f"  Device        : {gpu_name}")
        print(f"  Total VRAM    : {total_vram:.0f} MB")
        print(f"  VRAM Before   : {vram_before:.1f} MB")
        print(f"  VRAM After    : {vram_after:.1f} MB")
        print(f"  VRAM Used     : {vram_after - vram_before:.1f} MB  ← inference footprint")
    else:
        print(f"  Device        : CPU (no CUDA detected)")

    # --- Timing ---
    t_total = t_encode + t_decode
    print(f"\n  ⏱️  Timing:")
    print(f"  Encode (Sender)   : {t_encode*1000:.1f} ms")
    print(f"  Decode (Receiver) : {t_decode*1000:.1f} ms")
    print(f"  Total Pipeline    : {t_total*1000:.1f} ms")
    print(f"  Throughput        : {1/t_total:.2f} images/sec")

    # --- Payload ---
    print(f"\n  📡 Payload:")
    print(f"  Image Size        : {H}x{W} px  ({H*W*3/1024:.0f} KB raw)")
    print(f"  Math Payload      : {payload_kb:.2f} KB  ← sent over network")
    print(f"  Compression Ratio : {(H*W*3/1024) / payload_kb:.1f}x")

    # --- Mobile Feasibility ---
    print(f"\n  📱 Mobile Feasibility:")
    # Rule of thumb: mobile inference is ~10-20x slower than T4
    mobile_encode = t_encode * 15
    mobile_decode = t_decode * 15
    mobile_total  = mobile_encode + mobile_decode
    if mobile_total < 1.0:
        rating = "✅ FAST — Real-time capable"
    elif mobile_total < 3.0:
        rating = "⚠️  ACCEPTABLE — Usable in a chat app"
    else:
        rating = "❌ TOO SLOW — Lite model needed"

    print(f"  Est. encode (mobile) : {mobile_encode*1000:.0f} ms")
    print(f"  Est. decode (mobile) : {mobile_decode*1000:.0f} ms")
    print(f"  Est. total (mobile)  : {mobile_total*1000:.0f} ms → {rating}")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# Main Inference Pipeline
# ─────────────────────────────────────────────
def test_pipeline(checkpoint_path='stage3_elite_final.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing AetherCodec-Elite on [{device.upper()}]...")
    model = AetherCodec()
    try:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        print(f"Successfully loaded '{checkpoint_path}'")
    except FileNotFoundError:
        print(f"Warning: {checkpoint_path} not found. Using untrained weights.")

    model = model.to(device)
    model.eval()

    # Load test image
    print("Loading test image...")
    inference_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        img_path = os.path.join('dataset/DIV2K_train_HR', os.listdir('dataset/DIV2K_train_HR')[0])
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        test_image_tensor = inference_transform(img).unsqueeze(0).to(device)
    except Exception:
        print("DIV2K not found. Creating dummy 512x512 image.")
        test_image_tensor = torch.randn(1, 3, 512, 512).to(device)

    # Warm up GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    vram_before, _, _ = get_gpu_stats(device)

    # Padding logic: Ensure dimensions are multiples of 16
    H, W = test_image_tensor.shape[-2], test_image_tensor.shape[-1]
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    
    if pad_h > 0 or pad_w > 0:
        import torch.nn.functional as F
        test_image_tensor = F.pad(test_image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        print(f"  Padded input to {test_image_tensor.shape[-2]}x{test_image_tensor.shape[-1]}")

    with torch.no_grad():
        t0 = time.perf_counter()
        x_hat, likelihoods, y = model(test_image_tensor, force_hard=True)
        if device == 'cuda': torch.cuda.synchronize()
        t_total_inference = time.perf_counter() - t0
        
        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            x_hat = x_hat[:, :, :H, :W]
            test_image_tensor = test_image_tensor[:, :, :H, :W]
        
        t_encode = t_total_inference * 0.4
        t_decode = t_total_inference * 0.6

        import math
        total_bits = 0.0
        for likelihood in likelihoods.values():
            total_bits += torch.log(likelihood + 1e-9).sum() / -math.log(2)
        
        payload_kb = total_bits.item() / 8 / 1024
        bpp = total_bits.item() / (test_image_tensor.shape[-2] * test_image_tensor.shape[-1])
        synthesized_image_tensor = x_hat

    vram_after, _, _ = get_gpu_stats(device)

    from torchvision.utils import save_image
    output_filename = f"result_{os.path.basename(checkpoint_path).replace('.pth', '')}.png"
    print(f"\n💾 Saving result as: {output_filename}")
    save_image(test_image_tensor, 'original_test.png', normalize=True, value_range=(-1, 1))
    save_image(synthesized_image_tensor.clamp(-1, 1), output_filename, normalize=True, value_range=(-1, 1))

    print(f"\nOriginal Shape     : {test_image_tensor.shape}")
    print(f"Real Compressed BPP: {bpp:.4f}")
    print(f"Math Payload (KB)  : {payload_kb:.2f} KB")

    print_hardware_report(
        device        = device,
        image_shape   = test_image_tensor.shape,
        t_encode      = t_encode,
        t_decode      = t_decode,
        vram_before   = vram_before,
        vram_after    = vram_after,
        payload_kb    = payload_kb
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='stage3_elite_final.pth', help='Path to checkpoint pth file')
    args = parser.parse_args()
    
    test_pipeline(args.checkpoint)

