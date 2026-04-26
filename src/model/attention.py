import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    """
    Swin-style Window Attention Block.
    Input: (B, C, H, W)
    """
    def __init__(self, dim, window_size=8, num_heads=4, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.shape
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            
        # Pad if needed
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = nn.functional.pad(x, (0, pad_r, 0, pad_b))
            
        _, _, Hp, Wp = x.shape
        
        qkv = self.qkv(x) # (B, 3C, Hp, Wp)
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3) # (3, B, num_windows_h, num_windows_w, num_heads, window_size, window_size, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q.reshape(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads)
        k = k.reshape(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads)
        v = v.reshape(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.reshape(B, Hp // self.window_size, Wp // self.window_size, self.num_heads, self.window_size, self.window_size, C // self.num_heads)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        
        out = self.proj(out)
        
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]
            
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = WindowAttention(dim, window_size, num_heads, shift_size=0)
        self.norm2 = nn.GroupNorm(1, dim)
        self.attn_shifted = WindowAttention(dim, window_size, num_heads, shift_size=window_size//2)
        
        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )

    def forward(self, x):
        # Attention without shift
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        
        # Attention with shift
        x = x + self.attn_shifted(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return x
