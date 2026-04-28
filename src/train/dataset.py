import os
import urllib.request
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

def download_and_extract_div2k(root_dir="dataset"):
    """
    Downloads and extracts the DIV2K dataset automatically if it doesn't exist.
    """
    dataset_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = os.path.join(root_dir, "DIV2K_train_HR.zip")
    extract_path = os.path.join(root_dir, "DIV2K_train_HR")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print(f"Dataset already exists at {extract_path}. Skipping download.")
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    
    if not os.path.exists(zip_path):
        print(f"Downloading DIV2K dataset from {dataset_url} (approx 3.5GB)...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete!")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root_dir)
    print("Extraction complete!")
    
    return extract_path

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Expert Filter: Only include valid image files
        self.image_files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        self.transform = transform
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}. Error: {e}")
            # Fallback to another image if one is corrupted
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0

def get_dataloader(data_dir=None, batch_size=8, image_size=256, is_train=True):
    """
    Creates an Elite Dataloader for the AetherCodec engine.
    Ensures natural statistics are preserved for high-fidelity reconstruction.
    """
    if data_dir is None or data_dir == 'auto':
        data_dir = download_and_extract_div2k()
        
    if is_train:
        # ELITE TRAINING AUGMENTATIONS
        # We remove ColorJitter because we want the model to learn REAL color statistics.
        # We add Resize(image_size) as a safety guard before the RandomCrop.
        transform = transforms.Compose([
            transforms.Resize(image_size), # Safety Guard
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # Added for structural variance
            transforms.ToTensor(),
            # Correctly maps [0, 1] to [-1, 1] for Tanh decoder
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        # ELITE VALIDATION TRANSFORM
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    dataset = ImageFolderDataset(data_dir, transform=transform)
    
    # num_workers set to 2 for Colab's CPU environment to prevent freezes
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=2, 
        drop_last=True,
        pin_memory=True # Speed boost for CUDA
    )
    
    return dataloader
