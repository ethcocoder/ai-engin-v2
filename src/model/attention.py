import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

class WindowAttention(nn.Module):
    """
    Swin-style Window Attention with Relative Position Bias and Masking.
    Correctly implements spatial inductive bias for image reconstruction.
    """
    def __init__(self, dim, window_size=8, num_heads=4, shift_size=0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # FIX 1: Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        
        # Relative position index calculation
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Dynamic mask will be created in forward
        self.attn_mask = None
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.attention_weights = None
    
    def _create_shifted_mask(self, H, W, window_size, shift_size):
        """Cyclic shift masking to prevent wrap-around attention artifacts."""
        img_mask = torch.zeros((1, 1, H, W))
        
        h_slices = (slice(0, -window_size),
                   slice(-window_size, -shift_size),
                   slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                   slice(-window_size, -shift_size),
                   slice(-shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1
        
        mask_windows = img_mask.view(1, H // window_size, window_size,
                                      W // window_size, window_size)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4).contiguous()
        mask_windows = mask_windows.view(-1, window_size * window_size)
        
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        
        # Padding to window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = nn.functional.pad(x, (0, pad_r, 0, pad_b))
        
        _, _, Hp, Wp = x.shape
        
        # QKV Projection & Window Partition
        qkv = self.qkv(x)
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads,
                       Hp // self.window_size, self.window_size,
                       Wp // self.window_size, self.window_size)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q.reshape(-1, self.num_heads, self.window_size ** 2, C // self.num_heads)
        k = k.reshape(-1, self.num_heads, self.window_size ** 2, C // self.num_heads)
        v = v.reshape(-1, self.num_heads, self.window_size ** 2, C // self.num_heads)
        
        # Attention Calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # FIX 1: Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size ** 2, self.window_size ** 2, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # FIX 2: Dynamic shifted window mask based on H, W
        if self.shift_size > 0:
            # Check if cached mask is still valid for current resolution
            if not hasattr(self, 'cached_mask') or self.cached_mask.shape[0] != (Hp // self.window_size * Wp // self.window_size):
                self.cached_mask = self._create_shifted_mask(Hp, Wp, self.window_size, self.shift_size).to(x.device)
            
            nW = (Hp // self.window_size) * (Wp // self.window_size)
            attn = attn.view(B, nW, self.num_heads, self.window_size**2, self.window_size**2)
            attn = attn + self.cached_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, self.window_size**2, self.window_size**2)
        
        attn = attn.softmax(dim=-1)
        
        # Optional logging
        self.attention_weights = attn.detach().mean(1)
        
        # FIX 5: Dropout
        attn = self.attn_dropout(attn)
        
        # Merge windows
        out = (attn @ v)
        out = out.reshape(B, Hp // self.window_size, Wp // self.window_size,
                          self.num_heads, self.window_size, self.window_size,
                          C // self.num_heads)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        
        # Output Projection
        out = self.proj_dropout(self.proj(out))
        
        # Crop & Reverse Shift
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]
        
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        return out

class ChannelLayerNorm(nn.Module):
    """
    FIX 4: Native LayerNorm optimized for (B, C, H, W) tensors.
    Much faster than GroupNorm(1, C).
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2) # (B, C, H, W)

class SwinBlock(nn.Module):
    """
    Full Swin Transformer Block: Standard + Shifted Window Attention.
    """
    def __init__(self, dim, window_size=8, num_heads=4, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # FIX 3: Separate Norms for Path 1 & Path 2
        self.norm1 = ChannelLayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, shift_size=0, dropout=dropout)
        self.norm2 = ChannelLayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim*4, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.norm3 = ChannelLayerNorm(dim)
        self.attn_shifted = WindowAttention(dim, window_size, num_heads,
                                           shift_size=window_size//2, dropout=dropout)
        self.norm4 = ChannelLayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim*4, dim, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # FIX 8: Gradient Checkpointing
        if self.use_checkpoint and self.training:
            x = x + checkpoint(self.attn, self.norm1(x))
            x = x + checkpoint(self.mlp1, self.norm2(x))
            x = x + checkpoint(self.attn_shifted, self.norm3(x))
            x = x + checkpoint(self.mlp2, self.norm4(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp1(self.norm2(x))
            x = x + self.attn_shifted(self.norm3(x))
            x = x + self.mlp2(self.norm4(x))
        return x
