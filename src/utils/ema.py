import torch
import torch.nn as nn
from contextlib import contextmanager

class EMA:
    """
    Elite Exponential Moving Average (EMA) for Neural Codecs.
    Stabilizes training, especially for Adversarial (Stage 3) and 
    Quantization-aware training.
    
    Optimized for memory efficiency and buffer synchronization.
    """
    def __init__(self, model, decay=0.999, use_buffers=True):
        self.decay = decay
        self.use_buffers = use_buffers
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow copies
        # We include buffers (like GroupNorm stats or custom QVS caches) 
        # to ensure full model synchronization.
        for name, data in self._get_model_state(model).items():
            self.shadow[name] = data.clone().detach()

    def _get_model_state(self, model):
        """Helper to get both parameters and buffers."""
        state = {name: p.data for name, p in model.named_parameters() if p.requires_grad}
        if self.use_buffers:
            state.update({name: b.data for name, b in model.named_buffers()})
        return state

    @torch.no_grad()
    def update(self, model):
        """
        High-efficiency in-place update using Linear Interpolation (lerp).
        """
        for name, data in self._get_model_state(model).items():
            if name in self.shadow:
                # FIX: lerp only works on floating point tensors. 
                # For integer buffers (like Swin indices), we just copy.
                if torch.is_floating_point(data):
                    self.shadow[name].lerp_(data, 1.0 - self.decay)
                else:
                    self.shadow[name].copy_(data)

    def apply_shadow(self, model):
        """Overwrites model weights with EMA shadow weights."""
        model_state = self._get_model_state(model)
        for name, data in model_state.items():
            if name in self.shadow:
                self.backup[name] = data.clone()
                data.copy_(self.shadow[name])

    def restore(self, model):
        """Restores original weights from backup."""
        model_state = self._get_model_state(model)
        for name, data in model_state.items():
            if name in self.backup:
                data.copy_(self.backup[name])
        self.backup = {}

    @contextmanager
    def average_parameters(self, model):
        """
        Elite Context Manager for evaluation.
        Automatically applies and restores weights.
        
        Usage:
            with ema.average_parameters(model):
                evaluate(model)
        """
        self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'use_buffers': self.use_buffers
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.use_buffers = state_dict.get('use_buffers', True)
