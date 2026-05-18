import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

# Link to the isolated Peer Receiver logic
sys.path.append(os.path.abspath("peer_receiver"))
from model import LatentGenesisCore

def receiver_node():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[NODE: RECEIVER] Initiated. Listening for Aether Packets...")
    
    if not os.path.exists("transfer.pdox"):
        print("[NODE: RECEIVER] ERROR: No packet found on the mesh.")
        return

    # Load model from Peer Receiver directory
    model = LatentGenesisCore(latent_channels=16).to(device)
    checkpoint = torch.load("peer_receiver/universal_genesis_core.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Read the received packet
    import numpy as np
    z_raw = np.fromfile("transfer.pdox", dtype=np.float32).reshape(1, 16, 16, 16)
    z_q = torch.from_numpy(z_raw).to(device)

    print("[NODE: RECEIVER] Reconstructing HD Reality from Peer-A Packet...")
    with torch.no_grad():
        reconstructed = model.decoder(z_q)
    
    # Post-processing and Save
    reconstructed = torch.clamp(reconstructed * 0.5 + 0.5, 0, 1)
    recon_img = transforms.ToPILImage()(reconstructed.squeeze(0).cpu())
    recon_img.save("receiver_victory.png")
    
    print("[NODE: RECEIVER] SUCCESS: 'receiver_victory.png' synthesized.")

if __name__ == "__main__":
    receiver_node()
