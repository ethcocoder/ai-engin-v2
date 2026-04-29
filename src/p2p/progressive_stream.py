import torch
import torch.nn as nn

class ProgressiveBitstream:
    """
    Elite Progressive Streaming Engine.
    Implements coarse-to-fine layered encoding for low-latency P2P previews.
    """
    def __init__(self, num_layers=3):
        self.num_layers = num_layers

    def encode_progressive(self, y_hat):
        """
        Splits the quantized latent into multiple enhancement layers.
        Layer 0: Base (High step size, low bitrate)
        Layer 1+: Enhancements (Residuals)
        """
        layers = []
        residual = y_hat.clone()
        
        for i in range(self.num_layers):
            # Coarse-to-fine: each layer captures smaller residuals
            # e.g., for 3 layers: step multipliers 4, 2, 1
            step_multiplier = 2.0 ** (self.num_layers - i - 1)
            
            # Simple progressive quantization: round to the coarse grid
            coarse = torch.round(residual / step_multiplier) * step_multiplier
            layers.append(coarse)
            
            # Update residual for the next layer
            residual = residual - coarse
            
        return {
            'base': layers[0],                      # Layer 0
            'enhancements': layers[1:]              # Layers 1 to N
        }
    
    def decode_progressive(self, layers_dict, layers_to_use=None):
        """
        Reconstructs the latent up to the specified enhancement layer.
        """
        if layers_to_use is None:
            layers_to_use = self.num_layers
            
        y = layers_dict['base'].clone()
        
        # Add enhancements if requested and available
        for i in range(min(layers_to_use - 1, len(layers_dict['enhancements']))):
            y += layers_dict['enhancements'][i]
            
        return y
