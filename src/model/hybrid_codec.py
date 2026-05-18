"""
Aether-Blueprint: TiledHybridCodec

The master orchestrator that routes tiles between:
- Rule-Based path (polynomial surface fitting) for simple regions
- Neural path (AetherCodec) for complex regions

Then stitches them seamlessly via Gaussian alpha blending.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiler import TileManager
from .rules import ComplexityGate, PolynomialSurface
from .aether_codec import AetherCodec


class TiledHybridCodec(nn.Module):
    """
    Aether-Blueprint: Hybrid Rule-Based + Neural Image Codec.
    """
    def __init__(self, tile_size=128, overlap=16, complexity_threshold=50.0,
                 use_attention=True, use_hyperprior=True):
        super().__init__()
        
        self.tile_size = tile_size
        self.overlap = overlap
        self.padded_tile = tile_size + 2 * overlap  # 160
        
        # --- Module 1: Tile Manager ---
        self.tiler = TileManager(tile_size=tile_size, overlap=overlap)
        
        # --- Module 2: Decision Gate ---
        self.gate = ComplexityGate(threshold=complexity_threshold)
        
        # --- Module 3: Rule-Based Engine ---
        self.poly = PolynomialSurface(tile_size=self.padded_tile)
        
        # --- Module 4: Neural Engine ---
        self.neural = AetherCodec(
            use_attention=use_attention,
            use_hyperprior=use_hyperprior,
            use_checkpoint=False
        )
        
    def _pad_tile_for_neural(self, tile):
        """Pad tile to multiple of 64 for the neural encoder."""
        _, _, H, W = tile.shape
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        if pad_h > 0 or pad_w > 0:
            tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
        return tile, H, W
    
    def _crop_tile(self, tile, H, W):
        """Crop tile back to original size after neural decode."""
        return tile[:, :, :H, :W]
    
    def encode(self, image):
        """
        Encode a full image using the vectorized hybrid pipeline.
        """
        B = image.shape[0]
        tiles_stacked, grid = self.tiler.split(image) # (B, 16, 3, 160, 160)
        num_tiles = tiles_stacked.shape[1]
        
        # Step 2: Vectorized Classification
        decisions, scores = self.gate.decide(tiles_stacked) # (B, 16)
        
        # Step 3: Vectorized Encoding
        # Batch Polynomial Fitting
        coeffs_all, _ = self.poly.fit(tiles_stacked.reshape(-1, 3, self.padded_tile, self.padded_tile))
        
        neural_data = None
        complex_mask = decisions.view(-1).bool()
        if complex_mask.any():
            flat_tiles = tiles_stacked.reshape(-1, 3, self.padded_tile, self.padded_tile)
            complex_tiles = flat_tiles[complex_mask]
            tile_padded, _, _ = self._pad_tile_for_neural(complex_tiles)
            
            with torch.set_grad_enabled(self.training):
                # Mixed Precision Support (Level 2)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    x_hat_neu, likelihoods, metrics = self.neural(
                        tile_padded, 
                        hard_prob=1.0 if not self.training else None
                    )
            
            neural_data = {
                'metrics': metrics,
                'likelihoods': likelihoods,
                'mask': complex_mask
            }
            
        return {
            'grid': grid,
            'decisions': decisions,
            'scores': scores,
            'poly_coeffs_all': coeffs_all.view(B, num_tiles, 3, self.poly.num_coeffs),
            'neural_data': neural_data,
            'num_tiles': num_tiles,
        }
    
    def decode(self, encoding, device='cuda'):
        """
        Vectorized decode: Renders all tiles in two GPU passes.
        """
        grid = encoding['grid']
        decisions = encoding['decisions']
        B, num_tiles = decisions.shape
        coeffs_all = encoding['poly_coeffs_all']
        neural_data = encoding['neural_data']
        
        # 1. Render all Polynomial tiles
        decoded_tiles = self.poly.render(coeffs_all.reshape(B * num_tiles, 3, self.poly.num_coeffs), 
                                        size=self.padded_tile)
        decoded_tiles = decoded_tiles.view(B, num_tiles, 3, self.padded_tile, self.padded_tile)
        
        # 2. Overwrite Complex tiles with Neural decodes
        if neural_data is not None:
            complex_mask = neural_data['mask']
            y_hat = neural_data['metrics']['y_hat']
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    x_hat_neu = self.neural.decoder(y_hat, encoder_skips=None)
            
            x_hat_neu = self._crop_tile(x_hat_neu, self.padded_tile, self.padded_tile)
            
            # Place complex tiles back into the grid
            flat_decoded = decoded_tiles.view(B * num_tiles, 3, self.padded_tile, self.padded_tile)
            flat_decoded[complex_mask] = x_hat_neu
            decoded_tiles = flat_decoded.view(B, num_tiles, 3, self.padded_tile, self.padded_tile)
        
        # 3. Vectorized Stitch
        image = self.tiler.stitch(decoded_tiles, grid, channels=3)
        return torch.clamp(image, -1.0, 1.0)
    
    def forward(self, image, hard_prob=None):
        """Fast Forward Pass for Training."""
        encoding = self.encode(image)
        x_hat = self.decode(encoding, device=image.device)
        
        likelihoods = {}
        if encoding['neural_data']:
            likelihoods = encoding['neural_data']['likelihoods']
            
        num_complex = encoding['decisions'].sum().item()
        num_total = encoding['num_tiles'] * image.shape[0]
        
        metrics = {
            'decisions': encoding['decisions'],
            'num_poly_tiles': num_total - num_complex,
            'num_neural_tiles': num_complex,
            'poly_ratio': (num_total - num_complex) / num_total,
        }
        
        return x_hat, likelihoods, metrics
    
    def get_compression_stats(self, encoding):
        """
        Estimate the .bpox file size (Upgrade v2).
        """
        num_complex = encoding['decisions'].sum().item()
        num_total = encoding['num_tiles']
        num_poly = num_total - num_complex
        
        # Polynomial: num_coeffs × 3 channels × 4 bytes
        poly_bytes = num_poly * self.poly.num_coeffs * 3 * 4
        
        # Neural: estimate from latent size
        neural_bytes = 0
        if encoding['neural_data']:
            y_hat = encoding['neural_data']['metrics']['y_hat']
            # int16 + zlib ≈ 50% compression on quantized data
            neural_bytes += y_hat.numel() * 2 * 0.5
        
        header_bytes = 64 + 2
        
        return {
            'total_bytes': header_bytes + poly_bytes + int(neural_bytes),
            'poly_bytes': poly_bytes,
            'neural_bytes': int(neural_bytes),
            'num_poly': num_poly,
            'num_neural': num_complex,
        }
