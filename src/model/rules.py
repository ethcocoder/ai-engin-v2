"""
Aether-Blueprint: Rule-Based Encoding Module.

Contains:
- ComplexityGate: Decides if a tile is "simple" or "complex"
- PolynomialSurface: Fits/renders 2nd-degree polynomial surfaces
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexityGate(nn.Module):
    """
    The Decision Gate: Analyzes tile complexity to route between
    Rule-Based (polynomial) and Neural encoding.
    
    Uses Laplacian variance as the complexity metric:
    - Low variance = smooth/simple (sky, walls) → polynomial
    - High variance = textured/complex (faces, foliage) → neural
    """
    def __init__(self, threshold=50.0):
        super().__init__()
        self.threshold = threshold
        
        # Laplacian kernel (second derivative = edge detector)
        lap = torch.tensor([[0.,  1., 0.],
                            [1., -4., 1.],
                            [0.,  1., 0.]], dtype=torch.float32)
        # Expand for 3-channel convolution
        lap = lap.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer('laplacian', lap)
    
    @torch.no_grad()
    def analyze(self, tile):
        """
        Compute complexity score for a tile.
        
        Args:
            tile: (B, 3, H, W) tensor in [-1, 1]
        Returns:
            score: (B,) Laplacian variance per image in batch
        """
        # Apply Laplacian filter (groups=3 for per-channel)
        edges = F.conv2d(tile, self.laplacian, padding=1, groups=3)
        # Variance across spatial dims = complexity score
        score = edges.var(dim=[1, 2, 3])  # (B,)
        return score
    
    @torch.no_grad()
    def decide(self, tiles):
        """
        Classify each tile as simple (0) or complex (1).
        
        Args:
            tiles: list of (B, 3, H, W) tensors
        Returns:
            decisions: (B, num_tiles) tensor of 0s and 1s
            scores: (B, num_tiles) raw complexity scores
        """
        B = tiles[0].shape[0]
        num_tiles = len(tiles)
        
        scores = torch.zeros(B, num_tiles, device=tiles[0].device)
        decisions = torch.zeros(B, num_tiles, dtype=torch.long, device=tiles[0].device)
        
        for i, tile in enumerate(tiles):
            s = self.analyze(tile)
            scores[:, i] = s
            decisions[:, i] = (s > self.threshold).long()
        
        return decisions, scores


class PolynomialSurface(nn.Module):
    """
    2nd-Degree Polynomial Surface Fitting.
    
    Models a tile as: I(x, y) = ax² + by² + cxy + dx + ey + f
    
    For a simple tile (sky gradient, flat wall), this captures the
    content in just 6 floats per channel = 18 floats = 72 bytes total.
    Compare to neural: ~2-8 KB per tile.
    """
    def __init__(self, tile_size=160):
        super().__init__()
        self.tile_size = tile_size
        
        # Build design matrix and pseudoinverse for least-squares fitting
        A, pinv = self._build_matrices(tile_size)
        self.register_buffer('A', A)        # (N, 6)
        self.register_buffer('pinv', pinv)  # (6, N)
    
    def _build_matrices(self, size):
        """Build the polynomial design matrix and its pseudoinverse."""
        # Normalized coordinates [-1, 1]
        y = torch.linspace(-1, 1, size)
        x = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx.flatten()
        yy = yy.flatten()
        
        # Design matrix: each row = [x², y², xy, x, y, 1]
        A = torch.stack([
            xx ** 2,
            yy ** 2,
            xx * yy,
            xx,
            yy,
            torch.ones_like(xx)
        ], dim=1)  # (N, 6) where N = size*size
        
        # Pseudoinverse via SVD (stable for any tile size)
        pinv = torch.linalg.pinv(A)  # (6, N)
        
        return A, pinv
    
    @torch.no_grad()
    def fit(self, tile):
        """
        Fit polynomial to tile pixels.
        
        Args:
            tile: (B, 3, H, W) tensor
        Returns:
            coeffs: (B, 3, 6) polynomial coefficients
            residual_mse: (B,) fitting error (for quality monitoring)
        """
        B, C, H, W = tile.shape
        N = H * W
        
        # Handle size mismatch (recompute if needed)
        if H != self.tile_size or W != self.tile_size:
            A, pinv = self._build_matrices(H)
            A = A.to(tile.device)
            pinv = pinv.to(tile.device)
        else:
            A = self.A
            pinv = self.pinv
        
        # Flatten spatial dims: (B, 3, N)
        pixels = tile.reshape(B, C, N)
        
        # Least-squares: coeffs = pinv @ pixels
        # pinv: (6, N), pixels: (B, C, N) → coeffs: (B, C, 6)
        coeffs = torch.einsum('kn,bcn->bck', pinv, pixels)
        
        # Compute residual for quality monitoring
        fitted = torch.einsum('nk,bck->bcn', A, coeffs)  # (B, C, N)
        residual_mse = ((pixels - fitted) ** 2).mean(dim=[1, 2])  # (B,)
        
        return coeffs, residual_mse
    
    def render(self, coeffs, size=None):
        """
        Render a tile from polynomial coefficients.
        
        Args:
            coeffs: (B, 3, 6) polynomial coefficients
            size: output tile size (default: self.tile_size)
        Returns:
            tile: (B, 3, size, size) rendered surface
        """
        if size is None:
            size = self.tile_size
        
        if size != self.tile_size:
            A, _ = self._build_matrices(size)
            A = A.to(coeffs.device)
        else:
            A = self.A.to(coeffs.device)
        
        # pixels = A @ coeffs: (N, 6) @ (B, 3, 6)^T → (B, 3, N)
        pixels = torch.einsum('nk,bck->bcn', A, coeffs)
        
        # Clamp to valid range
        pixels = torch.clamp(pixels, -1.0, 1.0)
        
        return pixels.reshape(-1, 3, size, size)
