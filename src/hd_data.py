import os
import urllib.request
import sys
from PIL import Image
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
except ImportError:
    # Environment warning: This module requires Torch and Torchvision for execution
    # but the paths remain loadable for static analysis.
    pass

class CustomHDDataset(Dataset):
    """Loads any real HD images uploaded by the user."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0 # Dummy label since we don't need classification

def get_hd_dataloaders(image_dir='./hd_images', batch_size=4):
    """
    Creates a dataloader for HD 256x256 images.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    # Check if folder is empty, if so, dynamically pull HD images from the web
    existing_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing_images) == 0:
        print("[*] Folder is empty! Downloading 4 random 1024x1024 HD test images from the internet...")
        urls = [
            "https://picsum.photos/seed/telecom1/1024/1024",
            "https://picsum.photos/seed/telecom2/1024/1024",
            "https://picsum.photos/seed/telecom3/1024/1024",
            "https://picsum.photos/seed/telecom4/1024/1024"
        ]
        for i, url in enumerate(urls):
            try:
                file_path = os.path.join(image_dir, f"internet_hd_{i}.jpg")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"    [+] Downloaded web image to {file_path}")
            except Exception as e:
                print(f"    [!] Failed to download: {e}")

    # We resize to a standard crisp multiple of 8 for the ResNet architecture
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomHDDataset(image_dir, transform=transform)
    
    if len(dataset) == 0:
        print(f"[!] No images found in '{image_dir}'. Please upload some!")
        return None

    # For this testing scenario, train & test are the same to verify exact capacity
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
