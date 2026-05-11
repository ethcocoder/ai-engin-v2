import os
import requests
import zipfile
import tarfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

def download_file(url, destination):
    """
    Downloads a file with a progress bar using requests.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}")
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def download_and_extract_div2k(root_dir="dataset"):
    """
    Downloads and extracts the DIV2K dataset automatically.
    """
    dataset_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = os.path.join(root_dir, "DIV2K_train_HR.zip")
    extract_path = os.path.join(root_dir, "DIV2K_train_HR")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print(f"DIV2K already exists at {extract_path}. Skipping.")
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"Downloading DIV2K dataset...")
        download_file(dataset_url, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root_dir)
    return extract_path

def download_and_extract_clic(root_dir="dataset"):
    """
    Downloads and extracts the CLIC 2020 Professional training dataset.
    """
    dataset_url = "https://storage.googleapis.com/clic2020_public/p_datasets/train.zip"
    zip_path = os.path.join(root_dir, "CLIC_train.zip")
    extract_path = os.path.join(root_dir, "CLIC_train")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print(f"CLIC already exists at {extract_path}. Skipping.")
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"Downloading CLIC 2020 dataset...")
        download_file(dataset_url, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path

class MultiFolderDataset(Dataset):
    """
    Dataset that aggregates images from multiple root directories for superior generalization.
    """
    def __init__(self, root_dirs, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.image_files = []
        for root in root_dirs:
            if not os.path.exists(root):
                print(f"Warning: Path {root} does not exist. Skipping.")
                continue
                
            # Recursive search for images to handle nested CLIC structures
            for r, d, f in os.walk(root):
                for file in f:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.image_files.append(os.path.join(r, file))
        
        self.transform = transform
        print(f"Elite Dataset Initialized with {len(self.image_files)} images from {len(root_dirs)} sources.")
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in any of the directories: {root_dirs}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Skip corrupted images automatically
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0

def download_and_extract_coco_val(root_dir="dataset"):
    """
    Downloads and extracts the COCO 2017 Validation set (5,000 images).
    """
    dataset_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(root_dir, "coco_val2017.zip")
    extract_path = os.path.join(root_dir, "coco_val2017")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print(f"COCO Val already exists at {extract_path}. Skipping.")
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"Downloading COCO 2017 Validation set (~1GB)...")
        download_file(dataset_url, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path

def download_and_extract_flickr8k(root_dir="dataset"):
    """
    Downloads and extracts the Flickr8k dataset (8,000 images).
    """
    dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    zip_path = os.path.join(root_dir, "flickr8k.zip")
    extract_path = os.path.join(root_dir, "flickr8k")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print(f"Flickr8k already exists at {extract_path}. Skipping.")
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"Downloading Flickr8k dataset (~1GB)...")
        download_file(dataset_url, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path

def get_dataloader(data_dir=None, batch_size=8, image_size=256, is_train=True, use_clic=True, use_10k_plus=True):
    """
    Creates an Elite Dataloader for the AetherCodec engine.
    Now supports a massive mix of datasets (10,000+ samples) for maximum generalization.
    """
    data_paths = []
    
    if data_dir is None or data_dir == 'auto':
        print("🔧 Auto-Configuring Massive Foundation Training Data (Target: 10,000+ Samples)...")
        data_paths.append(download_and_extract_div2k()) # 800
        if use_clic:
            data_paths.append(download_and_extract_clic()) # ~1,600
        
        if use_10k_plus:
            data_paths.append(download_and_extract_coco_val()) # 5,000
            data_paths.append(download_and_extract_flickr8k()) # 8,000
            # Total: ~15,400 samples
    else:
        data_paths = [data_dir]
        
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(image_size + 32), # Resize slightly larger for better crop variety
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Added subtle jitter for robustness
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    dataset = MultiFolderDataset(data_paths, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=4 if torch.cuda.is_available() else 2, 
        drop_last=True,
        pin_memory=True
    )
    
    return dataloader
