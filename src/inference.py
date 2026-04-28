import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import time
import os
import argparse

from src.model.aether_codec import AetherCodec
from src.utils.entropy_coder import ParadoxEntropyCoder
from src.utils.metrics import psnr, ms_ssim

def get_gpu_stats():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def run_inference(image_path, model_path, device='cuda'):
    """
    Elite Inference Pipeline for AetherCodec-Elite.
    Simulates a real-world P2P transmission:
    Sender -> Encode -> Quantize -> .padox Binary -> Receiver -> Decompress -> Decode.
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"🚀 Initializing AetherCodec-Elite on {device}...")
    model = AetherCodec().to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Handle both raw state_dicts and training checkpoints
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"✅ Loaded Elite weights from {model_path}")
    else:
        print(f"⚠️  Warning: {model_path} not found. Using random weights.")
    
    model.eval()
    coder = ParadoxEntropyCoder()
    
    # 1. Load and Prepare Image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image {image_path} not found. Generating dummy...")
        x = torch.randn(1, 3, 512, 512).to(device)
    else:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        x = transform(img).unsqueeze(0).to(device)
    
    vram0 = get_gpu_stats()
    
    with torch.no_grad():
        # --- SENDER SIDE ---
        print("📤 Encoding...")
        t0 = time.perf_counter()
        
        # Expert Encode
        y, _ = model.encoder(x, return_skips=True)
        
        # Sovereign Quantization
        y_hat, y_step = model.y_quantizer(y, force_hard=True)
        
        # Hyperprior Analysis (On quantized y_hat for honesty)
        z_hat = torch.zeros(1, 1, 1, 1).to(device)
        z_step = torch.ones(1, 1, 1, 1).to(device)
        
        if model.use_hyperprior:
            z_hat, z_step, _ = model.hyperprior(y_hat, force_hard=True)
        
        # Binary Compression
        file_path = "output.padox"
        file_size = coder.compress(y_hat, z_hat, y_step, z_step, file_path)
        t_send = time.perf_counter() - t0
        
        # --- RECEIVER SIDE ---
        print("📥 Decoding...")
        t1 = time.perf_counter()
        
        # Honest Decompression (Reads metadata from .padox)
        y_recon, z_recon, y_step_recv, z_step_recv = coder.decompress(file_path, device=device)
        
        # Synthesis (Skips=None for true P2P simulation)
        x_hat = model.decoder(y_recon, encoder_skips=None)
        
        t_recv = time.perf_counter() - t1
        
    vram1 = get_gpu_stats()
    
    # Calculate Quality Metrics
    p_val = psnr(x, x_hat, data_range=2.0)
    m_val = ms_ssim(x, x_hat, data_range=2.0)
    
    print(f"\n" + "="*40)
    print(f"📊 ELITE INFERENCE RESULTS")
    print(f"="*40)
    print(f"  Source     : {image_path}")
    print(f"  File Size  : {file_size/1024:.2f} KB")
    print(f"  Send Time  : {t_send*1000:.1f} ms")
    print(f"  Recv Time  : {t_recv*1000:.1f} ms")
    print(f"  PSNR       : {p_val:.2f} dB")
    print(f"  MS-SSIM    : {m_val:.4f}")
    print(f"  VRAM Used  : {vram1 - vram0:.1f} MB")
    print(f"="*40 + "\n")
    
    # Save Result
    out_img = x_hat.clamp(-1, 1)
    save_image(out_img, "inference_result.png", normalize=True, value_range=(-1, 1))
    print(f"🖼️  Saved reconstruction to: inference_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elite AetherCodec Inference")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="stage3_elite_final.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    run_inference(args.image, args.model, args.device)
