"""
Aether-Blueprint: Tile Manager with Gaussian Overlap Blending.

Splits a 512x512 image into 4x4=16 tiles of 128x128.
Each tile is extracted with 16px overlap for seamless stitching.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TileManager(nn.Module):
    def __init__(self, tile_size=128, overlap=16):
        super().__init__()
        self.tile_size = tile_size
        self.overlap = overlap
        self.padded_tile = tile_size + 2 * overlap  # 160
        
        # Pre-compute the 2D Gaussian blending mask (160x160)
        self.register_buffer('_blend_mask', self._build_gaussian_mask())
    
    def _build_gaussian_mask(self):
        """
        Build a 2D Gaussian-tapered blending mask.
        Center 128x128 = 1.0, fades to 0.0 over the 16px border.
        """
        t = self.padded_tile  # 160
        o = self.overlap       # 16
        
        # 1D profile: flat 1.0 in center, Gaussian fade at edges
        w = torch.ones(t)
        
        # Left/top fade: cosine taper (smoother than Gaussian at boundary)
        fade = 0.5 * (1.0 + torch.cos(torch.linspace(math.pi, 0, o)))
        w[:o] = fade
        w[-o:] = fade.flip(0)
        
        # 2D = outer product
        mask = w.unsqueeze(1) * w.unsqueeze(0)  # (160, 160)
        return mask
    
    def get_grid_size(self, image_size):
        """Number of tiles per axis."""
        return image_size // self.tile_size
    
    def split(self, image):
        """
        Split image into overlapping tiles using vectorized 'unfold'.
        Returns stacked tensor: (B, num_tiles, C, 160, 160)
        """
        B, C, H, W = image.shape
        o = self.overlap
        s = self.tile_size
        p = self.padded_tile
        
        # 1. Vectorized Padding
        padded = F.pad(image, [o, o, o, o], mode='reflect')
        
        # 2. Vectorized Extraction using Unfold
        # Extract sliding windows of size 160x160 with stride 128
        # Output: (B, C * 160 * 160, num_tiles)
        tiles = F.unfold(padded, kernel_size=p, stride=s)
        
        # 3. Reshape to (B, num_tiles, C, 160, 160)
        num_tiles = tiles.shape[-1]
        tiles = tiles.view(B, C, p, p, num_tiles).permute(0, 4, 1, 2, 3)
        
        rows = H // s
        cols = W // s
        return tiles, (rows, cols)
    
    def stitch(self, tiles_stacked, grid, channels=3):
        """
        Vectorized reassembly using Gaussian alpha blending.
        Input: (B, num_tiles, C, 160, 160)
        """
        rows, cols = grid
        B, num_tiles, C, p, _ = tiles_stacked.shape
        device = tiles_stacked.device
        o = self.overlap
        s = self.tile_size
        H, W = rows * s, cols * s
        
        # Target canvas sizes
        pH, pW = H + 2 * o, W + 2 * o
        
        # Prepare Mask for Fold
        mask = self._blend_mask.to(device).view(1, 1, p, p).expand(B, 1, p, p)
        
        # Weight each tile by its Gaussian mask
        weighted_tiles = tiles_stacked * mask.unsqueeze(1) # (B, num_tiles, C, 160, 160)
        
        # Prepare for 'fold': (B, C * p * p, num_tiles)
        weighted_tiles = weighted_tiles.permute(0, 2, 3, 4, 1).reshape(B, C * p * p, num_tiles)
        weights_expanded = mask.unsqueeze(4).expand(B, 1, p, p, num_tiles).reshape(B, 1 * p * p, num_tiles)
        
        # Use F.fold to accumulate overlapping regions
        canvas = F.fold(weighted_tiles, output_size=(pH, pW), kernel_size=p, stride=s)
        weight_map = F.fold(weights_expanded, output_size=(pH, pW), kernel_size=p, stride=s)
        
        # Normalize and crop
        canvas = canvas / (weight_map + 1e-8)
        return canvas[:, :, o:o + H, o:o + W]
