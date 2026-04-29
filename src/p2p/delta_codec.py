import torch
import torch.nn as nn

class DeltaLatentCodec(nn.Module):
    """
    Elite Delta Latent Codec for P2P Streaming.
    Exploits temporal redundancy by encoding only the changes between consecutive frames.
    """
    def __init__(self, latent_dim=192, threshold=0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.threshold = threshold
        self.sequence_number = 0
        # Shared P2P state
        self.register_buffer('prev_latent', None)
    
    def encode_delta(self, y_current):
        """
        Encodes the current latent relative to the previous one.
        Returns payload with sequence and timing info.
        """
        self.sequence_number += 1
        import time
        
        if self.prev_latent is None:
            self.prev_latent = y_current.detach().clone()
            payload = {
                'seq': self.sequence_number,
                'ts': time.time(),
                'data': y_current,
                'is_delta': False
            }
            return payload
        
        delta = y_current - self.prev_latent
        # Sparse delta mask based on threshold
        mask = torch.abs(delta) > self.threshold
        sparse_delta = delta * mask.float()
        
        # Update shared state
        self.prev_latent = y_current.detach().clone()
        
        payload = {
            'seq': self.sequence_number,
            'ts': time.time(),
            'data': sparse_delta,
            'is_delta': True
        }
        return payload
    
    def decode_delta(self, delta_input, is_delta):
        """
        Reconstructs the current latent from the delta and previous state.
        """
        if not is_delta:
            self.prev_latent = delta_input.detach().clone()
            return delta_input
        
        if self.prev_latent is None:
            raise RuntimeError("Delta received but no previous latent state found.")
            
        y = self.prev_latent + delta_input
        self.prev_latent = y.detach().clone()
        
        return y

    def reset_session(self):
        """Resets the temporal state for a new P2P session."""
        self.prev_latent = None
