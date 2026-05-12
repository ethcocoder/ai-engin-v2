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
        Split image into overlapping tiles.
        
        Args:
            image: (B, C, H, W) tensor, H and W must be divisible by tile_size
        Returns:
            tiles: list of (B, C, 160, 160) tensors
            grid: (rows, cols) tuple
        """
        B, C, H, W = image.shape
        rows = H // self.tile_size
        cols = W // self.tile_size
        o = self.overlap
        
        # Pad image by overlap on all sides (reflect for natural edges)
        padded = F.pad(image, [o, o, o, o], mode='reflect')
        
        tiles = []
        for r in range(rows):
            for c in range(cols):
                r0 = r * self.tile_size
                c0 = c * self.tile_size
                tile = padded[:, :, r0:r0 + self.padded_tile,
                                    c0:c0 + self.padded_tile]
                tiles.append(tile)
        
        return tiles, (rows, cols)
    
    def stitch(self, tiles, grid, channels=3):
        """
        Reassemble tiles using Gaussian alpha blending.
        
        Args:
            tiles: list of (B, C, 160, 160) decoded tile tensors
            grid: (rows, cols)
        Returns:
            image: (B, C, H, W)
        """
        rows, cols = grid
        B = tiles[0].shape[0]
        device = tiles[0].device
        H = rows * self.tile_size
        W = cols * self.tile_size
        o = self.overlap
        
        # Accumulators on padded canvas
        pH, pW = H + 2 * o, W + 2 * o
        canvas = torch.zeros(B, channels, pH, pW, device=device)
        weight = torch.zeros(B, 1, pH, pW, device=device)
        
        mask = self._blend_mask.to(device)  # (160, 160)
        
        idx = 0
        for r in range(rows):
            for c in range(cols):
                r0 = r * self.tile_size
                c0 = c * self.tile_size
                tile = tiles[idx]
                
                canvas[:, :, r0:r0 + self.padded_tile,
                             c0:c0 + self.padded_tile] += tile * mask
                weight[:, :, r0:r0 + self.padded_tile,
                             c0:c0 + self.padded_tile] += mask
                idx += 1
        
        # Normalize and crop back to original size
        canvas = canvas / (weight + 1e-8)
        return canvas[:, :, o:o + H, o:o + W]
