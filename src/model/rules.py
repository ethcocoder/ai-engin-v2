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
    def decide(self, tiles):
        """
        Classify each tile as simple (0) or complex (1).
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
    3rd-Degree Polynomial Surface Fitting (Upgrade v2).
    
    Models a tile as: 
    I(x, y) = a + bx + cy + dx² + exy + fy² + gx³ + hx²y + ixy² + jy³
    
    10 coefficients per channel = 30 floats = 120 bytes total.
    Significant quality improvement over 2nd-degree for smooth gradients.
    """
    def __init__(self, tile_size=160):
        super().__init__()
        self.tile_size = tile_size
        self.num_coeffs = 10
        
        A, pinv = self._build_matrices(tile_size)
        self.register_buffer('A', A)        # (N, 10)
        self.register_buffer('pinv', pinv)  # (10, N)
    
    def _build_matrices(self, size):
        """Build the polynomial design matrix and its pseudoinverse."""
        y = torch.linspace(-1, 1, size)
        x = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx.flatten()
        yy = yy.flatten()
        
        # Design matrix: [1, x, y, x², xy, y², x³, x²y, xy², y³]
        A = torch.stack([
            torch.ones_like(xx),
            xx,
            yy,
            xx ** 2,
            xx * yy,
            yy ** 2,
            xx ** 3,
            (xx ** 2) * yy,
            xx * (yy ** 2),
            yy ** 3
        ], dim=1)  # (N, 10)
        
        pinv = torch.linalg.pinv(A)  # (10, N)
        return A, pinv
    
    @torch.no_grad()
    def fit(self, tile):
        """Fit 3rd-degree polynomial to tile pixels."""
        B, C, H, W = tile.shape
        N = H * W
        
        if H != self.tile_size or W != self.tile_size:
            A, pinv = self._build_matrices(H)
            A, pinv = A.to(tile.device), pinv.to(tile.device)
        else:
            A, pinv = self.A, self.pinv
        
        pixels = tile.reshape(B, C, N)
        
        # Least-squares: coeffs = pinv @ pixels
        # pinv: (10, N), pixels: (B, C, N) → coeffs: (B, C, 10)
        coeffs = torch.einsum('kn,bcn->bck', pinv, pixels)
        
        # Residual MSE for quality tracking
        fitted = torch.einsum('nk,bck->bcn', A, coeffs)
        residual_mse = ((pixels - fitted) ** 2).mean(dim=[1, 2])
        
        return coeffs, residual_mse
    
    def render(self, coeffs, size=None):
        """Render tile from 3rd-degree coefficients."""
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
