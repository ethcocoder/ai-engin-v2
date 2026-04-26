import os
import urllib.request
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        print(f"Downloading DIV2K dataset from {dataset_url}...")
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
        self.image_files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0

def get_dataloader(data_dir=None, batch_size=8, image_size=256, is_train=True):
    # Automatically download the dataset if data_dir is not provided or set to 'auto'
    if data_dir is None or data_dir == 'auto':
        data_dir = download_and_extract_div2k()
        
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
    dataset = ImageFolderDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2, drop_last=True)
    
    return dataloader
