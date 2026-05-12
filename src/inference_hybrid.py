import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import time
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.hybrid_codec import TiledHybridCodec
from src.utils.hybrid_coder import HybridEntropyCoder
from src.utils.metrics import psnr, ms_ssim

def run_hybrid_inference(image_path, model_path, threshold=50.0, device='cuda'):
    """
    Elite Aether-Blueprint Inference Pipeline.
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"🚀 Initializing Aether-Blueprint (Hybrid Codec) on {device}...")
    model = TiledHybridCodec(complexity_threshold=threshold).to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Load neural sub-model weights
        if 'state_dict' in state_dict:
            model.neural.load_state_dict(state_dict['state_dict'], strict=False)
        else:
            model.neural.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded Neural weights from {model_path}")
    else:
        print(f"⚠️  Warning: {model_path} not found. Neural path will use random weights.")
    
    model.eval()
    coder = HybridEntropyCoder()
    
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image {image_path} not found.")
        return

    img = Image.open(image_path).convert('RGB')
    orig_size = os.path.getsize(image_path)
    print(f"📁 Source: {image_path} ({orig_size/1024:.1f} KB, {img.size[0]}x{img.size[1]})")
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)), # Blueprint expects 512x512
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # --- SENDER SIDE ---
        print("📤 Hybrid Encoding...")
        t0 = time.perf_counter()
        
        encoding = model.encode(x)
        
        file_path = "output.bpox"
        file_size = coder.compress(encoding, file_path)
        t_send = time.perf_counter() - t0
        
        # Stats
        stats = model.get_compression_stats(encoding)
        
        # --- RECEIVER SIDE ---
        print("📥 Hybrid Decoding...")
        t1 = time.perf_counter()
        
        decoded_encoding = coder.decompress(file_path, device=device)
        x_hat = model.decode(decoded_encoding, device=device)
        
        t_recv = time.perf_counter() - t1
        
    # Quality Metrics
    p_val = psnr(x, x_hat, data_range=2.0)
    m_val = ms_ssim(x, x_hat, data_range=2.0)
    
    print(f"\n" + "="*50)
    print(f"📊 AETHER-BLUEPRINT RESULTS (Threshold: {threshold})")
    print(f"="*50)
    print(f"  Tiles (Total) : {encoding['num_tiles']} (4x4 grid)")
    print(f"  Math Tiles    : {stats['num_poly']} ({stats['num_poly']/16*100:.1f}%)")
    print(f"  Neural Tiles  : {stats['num_neural']} ({stats['num_neural']/16*100:.1f}%)")
    print(f"  .bpox Size    : {file_size/1024:.2f} KB")
    print(f"  Comp. Ratio   : {orig_size/file_size:.1f}x")
    print(f"  PSNR          : {p_val:.2f} dB")
    print(f"  MS-SSIM       : {m_val:.4f}")
    print(f"  Time (S/R)    : {t_send*1000:.1f}ms / {t_recv*1000:.1f}ms")
    print(f"="*50 + "\n")
    
    # Save Result
    out_img = x_hat.clamp(-1, 1)
    save_image(out_img, "blueprint_result.png", normalize=True, value_range=(-1, 1))
    print(f"🖼️  Saved hybrid reconstruction to: blueprint_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aether-Blueprint Hybrid Inference")
    parser.add_argument("--image", type=str, default="test.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="stage1_foundation.pth", help="Path to neural weights")
    parser.add_argument("--threshold", type=float, default=50.0, help="Complexity threshold (lower = more neural)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    run_hybrid_inference(args.image, args.model, args.threshold, args.device)
