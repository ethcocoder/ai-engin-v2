import torch
import sys
from pathlib import Path
sys.path.append(str(Path("src").resolve()))
from model import LatentGenesisCore
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import os

model = LatentGenesisCore(16)
checkpoint = torch.load('checkpoints/universal_tpu_master.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

url = "https://picsum.photos/seed/42/1024/1024"
urllib.request.urlretrieve(url, "test.jpg")
img = Image.open("test.jpg").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
x = transform(img).unsqueeze(0)

with torch.no_grad():
    mu, logvar = model.encoder(x)
    
    # 1. Direct mu
    rec_mu = model.decoder(mu)
    
    # 2. Quantized mu (demo_hd.py approach)
    z_q = torch.round(torch.clamp(mu, -1, 1) * 127.5) / 127.5
    rec_q = model.decoder(z_q)
    
    # 3. Model forward
    rec_fwd, _, _ = model(x)

def psnr(a, b):
    a = torch.clamp(a * 0.5 + 0.5, 0, 1)
    b = torch.clamp(b * 0.5 + 0.5, 0, 1)
    mse = torch.mean((a - b)**2).item()
    if mse == 0: return 100
    return 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()

print(f"PSNR with   mu: {psnr(x, rec_mu):.2f}")
print(f"PSNR with  z_q: {psnr(x, rec_q):.2f}")
print(f"PSNR with eval: {psnr(x, rec_fwd):.2f}")
