import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

# Link to the isolated Peer Sender logic
sys.path.append(os.path.abspath("peer_sender"))
from model import LatentGenesisCore

def sender_node():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[NODE: SENDER] Initiated. Loading Peer-A Neural Weights...")
    
    # Load model from Peer Sender directory
    model = LatentGenesisCore(latent_channels=16).to(device)
    checkpoint = torch.load("peer_sender/universal_genesis_core.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load 1080p Image
    img_path = "test_local/johan.png"
    img = Image.open(img_path).convert('RGB')
    
    # Neural Pre-processing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    print(f"[NODE: SENDER] Compressing '{img_path}' to Aether Packet...")
    with torch.no_grad():
        mu, _ = model.encoder(tensor)
        # Apply 8-bit Quantization
        z_q = torch.round(torch.clamp(mu, -1, 1) * 127.5) / 127.5
    
    # Save the 'transfer.pdox' packet
    z_q.cpu().numpy().tofile("transfer.pdox")
    print(f"[NODE: SENDER] PACKET BROADCASTED: 'transfer.pdox' ({os.path.getsize('transfer.pdox') / 1024:.1f} KB)")

if __name__ == "__main__":
    sender_node()
