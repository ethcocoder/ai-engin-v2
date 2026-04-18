"""
data.py — Paradox Genesis Universal HD Data Pipeline
=====================================================
Downloads and prepares the STL-10 dataset (100,000 unlabeled images) 
for Universal Neural Compression training. 

Images are 96x96 (HD-base), providing much higher frequency detail 
than CIFAR-10, allowing the model to learn universal textures.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import torch

def get_dataloaders(
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 4,
    pin_memory: bool = True,
    use_hd: bool = True,
    sample_limit: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return DataLoaders. 
    If use_hd=True, use STL-10 (100,000 images, 96x96).
    If use_hd=False, use CIFAR-10 (60,000 images, 32x32).
    """
    
    # Advanced Augmentation to learn "Universal Pattern Grammar"
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256 if use_hd else 32, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256 if use_hd else 32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if use_hd:
        trainset = torchvision.datasets.STL10(root=root, split='unlabeled', download=True, transform=train_transform)
        testset = torchvision.datasets.STL10(root=root, split='test', download=True, transform=test_transform)
        
        # Fast-Pattern Logic: Subset the 100k images
        if sample_limit and sample_limit < len(trainset):
            import torch.utils.data as data
            indices = torch.randperm(len(trainset))[:sample_limit]
            trainset = data.Subset(trainset, indices)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )

    _persistent = num_workers > 0
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=_persistent
    )

    return trainloader, testloader
