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
    
    Flow:
        Image → Split into 16 tiles → Classify each tile →
        Simple tiles → Polynomial fit (72 bytes each)
        Complex tiles → Neural encode (variable KB each)
        → Pack into .padox → Transmit →
        → Unpack → Render poly / Decode neural →
        → Gaussian stitch → Output Image
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
        Encode a full image using the hybrid pipeline.
        
        Args:
            image: (B, 3, H, W) tensor in [-1, 1]
        Returns:
            encoding: dict with all data needed for decoding and .padox packing
        """
        B = image.shape[0]
        
        # Step 1: Split into tiles
        tiles, grid = self.tiler.split(image)
        num_tiles = len(tiles)
        
        # Step 2: Classify each tile
        decisions, scores = self.gate.decide(tiles)
        
        # Step 3: Encode each tile via its assigned path
        poly_coeffs = {}      # tile_idx → (B, 3, 6)
        poly_residuals = {}   # tile_idx → (B,)
        neural_latents = {}   # tile_idx → {y_hat, y_step, z_hat, z_step, ...}
        neural_likelihoods = {}
        neural_metrics = {}
        
        for i, tile in enumerate(tiles):
            # Check if ALL items in batch agree on decision
            # (For simplicity, use majority vote)
            is_complex = decisions[:, i].float().mean() > 0.5
            
            if not is_complex:
                # --- RULE-BASED PATH ---
                coeffs, residual = self.poly.fit(tile)
                poly_coeffs[i] = coeffs
                poly_residuals[i] = residual
            else:
                # --- NEURAL PATH ---
                tile_padded, orig_h, orig_w = self._pad_tile_for_neural(tile)
                
                with torch.set_grad_enabled(self.training):
                    x_hat, likelihoods, metrics = self.neural(
                        tile_padded, 
                        hard_prob=1.0 if not self.training else None
                    )
                
                x_hat = self._crop_tile(x_hat, orig_h, orig_w)
                
                neural_latents[i] = {
                    'y_hat': metrics['y_hat'],
                    'y_step': metrics['y_step'],
                    'z_hat': metrics.get('z_hat'),
                    'z_step': metrics.get('z_step'),
                    'orig_h': orig_h,
                    'orig_w': orig_w,
                }
                neural_likelihoods[i] = likelihoods
                neural_metrics[i] = metrics
        
        return {
            'grid': grid,
            'decisions': decisions,
            'scores': scores,
            'poly_coeffs': poly_coeffs,
            'poly_residuals': poly_residuals,
            'neural_latents': neural_latents,
            'neural_likelihoods': neural_likelihoods,
            'neural_metrics': neural_metrics,
            'num_tiles': num_tiles,
        }
    
    def decode(self, encoding, device='cuda'):
        """
        Decode from the hybrid encoding back to a full image.
        
        Args:
            encoding: dict from encode() or from .padox decompression
        Returns:
            image: (B, 3, H, W) reconstructed image
        """
        grid = encoding['grid']
        decisions = encoding['decisions']
        poly_coeffs = encoding['poly_coeffs']
        neural_latents = encoding['neural_latents']
        num_tiles = encoding['num_tiles']
        
        decoded_tiles = []
        
        for i in range(num_tiles):
            is_complex = decisions[:, i].float().mean() > 0.5
            
            if not is_complex and i in poly_coeffs:
                # --- RULE-BASED DECODE ---
                tile = self.poly.render(poly_coeffs[i], size=self.padded_tile)
                decoded_tiles.append(tile)
            elif i in neural_latents:
                # --- NEURAL DECODE ---
                lat = neural_latents[i]
                y_hat = lat['y_hat']
                
                with torch.no_grad():
                    x_hat = self.neural.decoder(y_hat, encoder_skips=None)
                
                x_hat = self._crop_tile(x_hat, lat['orig_h'], lat['orig_w'])
                decoded_tiles.append(x_hat)
            else:
                # Fallback: black tile
                B = decisions.shape[0]
                decoded_tiles.append(
                    torch.zeros(B, 3, self.padded_tile, self.padded_tile, device=device)
                )
        
        # Step 4: Stitch with Gaussian blending
        image = self.tiler.stitch(decoded_tiles, grid, channels=3)
        return torch.clamp(image, -1.0, 1.0)
    
    def forward(self, image, hard_prob=None):
        """
        Full forward pass for training.
        Returns reconstructed image, likelihoods, and metrics.
        """
        encoding = self.encode(image)
        
        # Decode all tiles
        grid = encoding['grid']
        decisions = encoding['decisions']
        decoded_tiles = []
        
        all_likelihoods = {'y': [], 'z': []}
        
        for i in range(encoding['num_tiles']):
            is_complex = decisions[:, i].float().mean() > 0.5
            
            if not is_complex and i in encoding['poly_coeffs']:
                tile = self.poly.render(encoding['poly_coeffs'][i], 
                                        size=self.padded_tile)
                decoded_tiles.append(tile)
            elif i in encoding['neural_latents']:
                lat = encoding['neural_latents'][i]
                x_hat = self.neural.decoder(lat['y_hat'], encoder_skips=None)
                x_hat = self._crop_tile(x_hat, lat['orig_h'], lat['orig_w'])
                decoded_tiles.append(x_hat)
                
                # Collect likelihoods for rate computation
                if i in encoding['neural_likelihoods']:
                    lk = encoding['neural_likelihoods'][i]
                    if 'y' in lk:
                        all_likelihoods['y'].append(lk['y'])
                    if 'z' in lk:
                        all_likelihoods['z'].append(lk['z'])
            else:
                B = image.shape[0]
                decoded_tiles.append(
                    torch.zeros(B, 3, self.padded_tile, self.padded_tile, 
                               device=image.device)
                )
        
        x_hat = self.tiler.stitch(decoded_tiles, grid, channels=3)
        x_hat = torch.clamp(x_hat, -1.0, 1.0)
        
        # Merge likelihoods from all neural tiles
        merged_likelihoods = {}
        if all_likelihoods['y']:
            merged_likelihoods['y'] = torch.cat(all_likelihoods['y'], dim=0)
        if all_likelihoods['z']:
            merged_likelihoods['z'] = torch.cat(all_likelihoods['z'], dim=0)
        
        # Build metrics
        num_poly = len(encoding['poly_coeffs'])
        num_neural = len(encoding['neural_latents'])
        
        metrics = {
            'decisions': decisions,
            'scores': encoding['scores'],
            'num_poly_tiles': num_poly,
            'num_neural_tiles': num_neural,
            'poly_ratio': num_poly / max(1, encoding['num_tiles']),
        }
        
        # Add poly residuals to metrics
        if encoding['poly_residuals']:
            avg_residual = torch.stack(list(encoding['poly_residuals'].values())).mean()
            metrics['poly_residual_mse'] = avg_residual
        
        return x_hat, merged_likelihoods, metrics
    
    def get_compression_stats(self, encoding):
        """
        Estimate the .padox file size for this encoding.
        """
        num_poly = len(encoding['poly_coeffs'])
        num_neural = len(encoding['neural_latents'])
        
        # Polynomial: 6 coeffs × 3 channels × 4 bytes = 72 bytes per tile
        poly_bytes = num_poly * 72
        
        # Neural: estimate from latent size
        neural_bytes = 0
        for lat in encoding['neural_latents'].values():
            y_hat = lat['y_hat']
            # int16 + zlib ≈ 50% compression on quantized data
            neural_bytes += y_hat.numel() * 2 * 0.5
            if lat['z_hat'] is not None:
                neural_bytes += lat['z_hat'].numel() * 2 * 0.5
        
        # Header + decision map
        header_bytes = 64 + 2
        
        return {
            'total_bytes': header_bytes + poly_bytes + int(neural_bytes),
            'poly_bytes': poly_bytes,
            'neural_bytes': int(neural_bytes),
            'header_bytes': header_bytes,
            'num_poly': num_poly,
            'num_neural': num_neural,
        }
