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
    Elite Decision Gate: Routes tiles between Rule-Based and Neural paths.
    
    Uses Sobel Edge Variance for high-sensitivity detail detection:
    - Low variance = simple (sky, smooth gradients) → polynomial
    - High variance = complex (textures, edges, text) → neural
    """
    def __init__(self, threshold=50.0):
        super().__init__()
        self.threshold = threshold
        
        # Sobel Kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    @torch.no_grad()
    def analyze(self, tile):
        """
        Compute complexity score using Sobel magnitude variance.
        """
        # Grouped convolution for RGB edges
        sx = F.conv2d(tile, self.sobel_x, padding=1, groups=3)
        sy = F.conv2d(tile, self.sobel_y, padding=1, groups=3)
        
        # Magnitude squared (faster than sqrt)
        magnitude = sx**2 + sy**2
        
        # Variance of the edge magnitude captures "structured complexity"
        score = magnitude.var(dim=[1, 2, 3])
        return score
    
    @torch.no_grad()
    def decide(self, tiles_stacked):
        """
        Classify all tiles in a batch as simple (0) or complex (1).
        Input: (B, num_tiles, 3, 160, 160)
        """
        B, num_tiles, C, H, W = tiles_stacked.shape
        
        # Flatten B and num_tiles to process all tiles at once: (B * num_tiles, 3, 160, 160)
        flat_tiles = tiles_stacked.reshape(B * num_tiles, C, H, W)
        
        # Vectorized Analysis
        scores = self.analyze(flat_tiles) # (B * num_tiles,)
        
        # Reshape back to (B, num_tiles)
        scores = scores.view(B, num_tiles)
        decisions = (scores > self.threshold).long()
        
        return decisions, scores


class PolynomialSurface(nn.Module):
    """
    Elite Polynomial Surface Fitting (Upgrade v3 - Chebyshev Basis).
    
    Dynamically generates Chebyshev polynomials up to a specified degree to prevent
    Runge's phenomenon (edge oscillations) and ensure absolute mathematical perfection.
    """
    def __init__(self, tile_size=160, degree=8):
        super().__init__()
        self.tile_size = tile_size
        self.degree = degree
        # Number of coefficients for 2D polynomial of degree D is (D+1)*(D+2)/2
        self.num_coeffs = (degree + 1) * (degree + 2) // 2
        
        A, pinv = self._build_matrices(tile_size)
        self.register_buffer('A', A)        # (N, num_coeffs)
        self.register_buffer('pinv', pinv)  # (num_coeffs, N)
        
    def chebyshev(self, n, x):
        """Evaluate Chebyshev polynomial of degree n at x using recurrence."""
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        
        t0 = torch.ones_like(x)
        t1 = x
        for _ in range(2, n + 1):
            t2 = 2 * x * t1 - t0
            t0 = t1
            t1 = t2
        return t2
    
    def _build_matrices(self, size):
        y = torch.linspace(-1, 1, size)
        x = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx.flatten()
        yy = yy.flatten()
        
        terms = []
        for d in range(self.degree + 1):
            for i in range(d + 1):
                j = d - i
                terms.append(self.chebyshev(i, xx) * self.chebyshev(j, yy))
                
        A = torch.stack(terms, dim=1) # (N, num_coeffs)
        pinv = torch.linalg.pinv(A)   # (num_coeffs, N)
        return A, pinv
        
    @torch.no_grad()
    def fit(self, tile):
        B, C, H, W = tile.shape
        N = H * W
        
        if H != self.tile_size or W != self.tile_size:
            A, pinv = self._build_matrices(H)
            A, pinv = A.to(tile.device), pinv.to(tile.device)
        else:
            A, pinv = self.A, self.pinv
            
        pixels = tile.reshape(B, C, N)
        
        # Least-squares: coeffs = pinv @ pixels
        coeffs = torch.einsum('kn,bcn->bck', pinv, pixels)
        
        # Residual MSE for quality tracking
        fitted = torch.einsum('nk,bck->bcn', A, coeffs)
        residual_mse = ((pixels - fitted) ** 2).mean(dim=[1, 2])
        
        return coeffs, residual_mse

    def render(self, coeffs, size=None):
        if size is None:
            size = self.tile_size
            
        if size != self.tile_size:
            A, _ = self._build_matrices(size)
            A = A.to(coeffs.device)
        else:
            A = self.A.to(coeffs.device)
            
        # Render: pixels = A @ coeffs
        pixels = torch.einsum('nk,bck->bcn', A, coeffs)
        
        # Stability clamp
        pixels = torch.clamp(pixels, -1.0, 1.0)
        
        return pixels.reshape(-1, 3, size, size)
