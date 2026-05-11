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
    Downloads a file with a progress bar and safety checks.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        # Expert Guard: Prevent unzipping small HTML error pages as datasets
        if total_size < 10000:
             raise RuntimeError(f"Server returned a suspiciously small file ({total_size} bytes). Likely a 404/403 landing page.")

        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}")
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination) # Clean up partial downloads
        raise e

def download_and_extract_div2k(root_dir="dataset"):
    """
    Downloads and extracts the DIV2K dataset automatically.
    """
    dataset_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = os.path.join(root_dir, "DIV2K_train_HR.zip")
    extract_path = os.path.join(root_dir, "DIV2K_train_HR")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    try:
        if not os.path.exists(zip_path):
            download_file(dataset_url, zip_path)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        return extract_path
    except Exception as e:
        if os.path.exists(zip_path): os.remove(zip_path)
        raise RuntimeError(f"DIV2K Download/Extract failed: {e}")

def download_and_extract_clic(root_dir="dataset"):
    """
    Downloads and extracts the CLIC 2020 Professional training dataset.
    """
    dataset_url = "https://storage.googleapis.com/clic2020_public/p_datasets/train.zip"
    zip_path = os.path.join(root_dir, "CLIC_train.zip")
    extract_path = os.path.join(root_dir, "CLIC_train")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    try:
        if not os.path.exists(zip_path):
            download_file(dataset_url, zip_path)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return extract_path
    except Exception as e:
        if os.path.exists(zip_path): os.remove(zip_path)
        raise RuntimeError(f"CLIC Download/Extract failed: {e}")

def download_and_extract_coco_val(root_dir="dataset"):
    """
    Downloads and extracts the COCO 2017 Validation set (5,000 images).
    """
    dataset_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(root_dir, "coco_val2017.zip")
    extract_path = os.path.join(root_dir, "coco_val2017")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    try:
        if not os.path.exists(zip_path):
            download_file(dataset_url, zip_path)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return extract_path
    except Exception as e:
        if os.path.exists(zip_path): os.remove(zip_path)
        raise RuntimeError(f"COCO Download/Extract failed: {e}")

def download_and_extract_flickr8k(root_dir="dataset"):
    """
    Downloads and extracts the Flickr8k dataset (8,000 images).
    """
    dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    zip_path = os.path.join(root_dir, "flickr8k.zip")
    extract_path = os.path.join(root_dir, "flickr8k")

    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        return extract_path

    os.makedirs(root_dir, exist_ok=True)
    try:
        if not os.path.exists(zip_path):
            download_file(dataset_url, zip_path)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return extract_path
    except Exception as e:
        if os.path.exists(zip_path): os.remove(zip_path)
        raise RuntimeError(f"Flickr8k Download/Extract failed: {e}")

class MultiFolderDataset(Dataset):
    """
    Dataset that aggregates images from multiple root directories for superior generalization.
    """
    def __init__(self, root_dirs, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.image_files = []
        for root in root_dirs:
            if root is None or not os.path.exists(root):
                print(f"Warning: Path {root} does not exist or is invalid. Skipping.")
                continue
                
            # Recursive search for images
            for r, d, f in os.walk(root):
                # Expert Filter: Skip macOS metadata folders
                if '__MACOSX' in r:
                    continue
                    
                for file in f:
                    # Filter: Only valid images, skip hidden system files (like ._ file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not file.startswith('._'):
                        self.image_files.append(os.path.join(r, file))
        
        self.transform = transform
        print(f"Elite Dataset Initialized with {len(self.image_files)} images from {len(root_dirs)} sources.")
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in any of the directories: {root_dirs}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            current_idx = (idx + attempt) % len(self)
            img_path = self.image_files[current_idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, 0
            except Exception as e:
                # Log error and try next image
                attempt += 1
                
        raise RuntimeError(f"Failed to load a valid image after {max_attempts} attempts. Last error: {e}")

def get_dataloader(data_dir=None, batch_size=8, image_size=256, is_train=True, use_clic=True, use_10k_plus=True):
    """
    Creates an Elite Dataloader for the AetherCodec engine.
    Now supports a massive mix of datasets (10,000+ samples) for maximum generalization.
    """
    data_paths = []
    
    if data_dir is None or data_dir == 'auto':
        print("🔧 Auto-Configuring Massive Foundation Training Data (Target: 10,000+ Samples)...")
        
        # Mandatory: DIV2K (Foundation)
        try:
            data_paths.append(download_and_extract_div2k())
        except Exception as e:
            print(f"❌ Critical Error: Could not setup DIV2K: {e}")
            raise e

        # Optional High-Fidelity sources (Graceful skips if server down)
        if use_clic:
            try:
                data_paths.append(download_and_extract_clic())
            except Exception as e:
                print(f"⚠️ Warning: Skipping CLIC dataset: {e}")
        
        if use_10k_plus:
            try:
                data_paths.append(download_and_extract_coco_val())
            except Exception as e:
                print(f"⚠️ Warning: Skipping COCO dataset: {e}")
            
            try:
                data_paths.append(download_and_extract_flickr8k())
            except Exception as e:
                print(f"⚠️ Warning: Skipping Flickr8k dataset: {e}")
                
        # Final Verification
        if not any(data_paths):
            raise RuntimeError("No datasets could be initialized.")
    else:
        data_paths = [data_dir]
        
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
