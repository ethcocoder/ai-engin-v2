import torch
import numpy as np
import zlib
import struct
import os

class ParadoxEntropyCoder:
    """
    Elite Entropy Coder: The Gateway to the .padox Binary Format.
    
    This coder bridges the gap between neural tensors and actual disk space.
    It handles quantization-aware serialization, metadata preservation (steps/shapes),
    and high-efficiency payload compression.
    
    CRITICAL FIX v6: Proper quantize-before-serialize pipeline.
    The latent y_hat is a FLOAT tensor. We must convert it to integer symbols
    by dividing by step_size, THEN cast to int16 for storage. On decompress,
    we read the int16 symbols and multiply by step_size to recover floats.
    """
    def __init__(self, magic_bytes=b"PDOX", version=6):
        self.magic_bytes = magic_bytes
        self.version = version

    def compress(self, y_hat, z_hat, y_step, z_step, output_path):
        """
        Compresses the neural latents and their quantization metadata.
        
        y_hat: Quantized main latent (B, C, H, W) — float values on step grid
        z_hat: Quantized hyper-latent (B, C, H, W) — float values on step grid
        y_step: Per-channel quantization step for y (C,) or (1, C, 1, 1)
        z_step: Per-channel quantization step for z (C,) or (1, C, 1, 1)
        """
        # 1. Convert step tensors to flat numpy arrays
        y_step_np = y_step.detach().cpu().view(-1).numpy().astype(np.float32)
        z_step_np = z_step.detach().cpu().view(-1).numpy().astype(np.float32)
        
        # 2. CRITICAL: Convert float latents to integer symbols
        #    y_hat is already quantized to the step grid, so y_hat / step gives integers
        #    We expand step to match spatial dims: (1, C, 1, 1)
        y_step_expand = y_step.detach().cpu().view(1, -1, 1, 1).float()
        z_step_expand = z_step.detach().cpu().view(1, -1, 1, 1).float()
        
        # Clamp step to avoid division by zero
        y_step_expand = torch.clamp(y_step_expand, min=1e-6)
        z_step_expand = torch.clamp(z_step_expand, min=1e-6)
        
        # Divide by step to get integer symbols, then round for safety
        y_symbols = torch.round(y_hat.detach().cpu().float() / y_step_expand)
        z_symbols = torch.round(z_hat.detach().cpu().float() / z_step_expand)
        
        # Convert to int16 (supports range [-32768, 32767])
        y_int = y_symbols.numpy().astype(np.int16)
        z_int = z_symbols.numpy().astype(np.int16)
        
        # 3. Serialize shapes
        b, cy, hy, wy = y_int.shape
        _, cz, hz, wz = z_int.shape
        
        # 4. Pack Payloads with zlib (production: torchac arithmetic coding)
        y_payload = zlib.compress(y_int.tobytes(), level=9)
        z_payload = zlib.compress(z_int.tobytes(), level=9)
        
        # 5. Build Header (Magic, Ver, B, CY, HY, WY, CZ, HZ, WZ, LenY, LenZ)
        header_base = struct.pack(">4sIIIIIIIIII", 
                             self.magic_bytes, self.version,
                             b, cy, hy, wy, cz, hz, wz,
                             len(y_payload), len(z_payload))
        
        y_step_data = y_step_np.tobytes()
        z_step_data = z_step_np.tobytes()
        
        # 6. Write Atomic .padox File
        with open(output_path, 'wb') as f:
            f.write(header_base)
            f.write(y_step_data)  # [CY * 4 bytes]
            f.write(z_step_data)  # [CZ * 4 bytes]
            f.write(y_payload)
            f.write(z_payload)
            
        return os.path.getsize(output_path)

    def decompress(self, input_path, device='cuda'):
        """
        The 'Honest' Decoder's Entry Point.
        Reconstructs floating-point latents from a binary bitstream.
        """
        with open(input_path, 'rb') as f:
            # 1. Read Base Header
            header_size = struct.calcsize(">4sIIIIIIIIII")
            header_data = f.read(header_size)
            magic, ver, b, cy, hy, wy, cz, hz, wz, len_y, len_z = struct.unpack(">4sIIIIIIIIII", header_data)
            
            if magic != self.magic_bytes:
                raise ValueError(f"Invalid file signature: {magic}")
                
            # 2. Read Quantization Metadata (Steps)
            y_step_data = f.read(cy * 4)
            z_step_data = f.read(cz * 4)
            
            y_step_arr = np.frombuffer(y_step_data, dtype=np.float32)
            z_step_arr = np.frombuffer(z_step_data, dtype=np.float32)
            
            # 3. Read and Decompress Payloads
            y_payload = f.read(len_y)
            z_payload = f.read(len_z)
            
            y_int_data = zlib.decompress(y_payload)
            z_int_data = zlib.decompress(z_payload)
            
            # 4. Reconstruct Integer Symbol Tensors
            y_int = np.frombuffer(y_int_data, dtype=np.int16).reshape(b, cy, hy, wy)
            z_int = np.frombuffer(z_int_data, dtype=np.int16).reshape(b, cz, hz, wz)
            
            # 5. RESCALE: Multiply integer symbols by step to recover float latents
            y_symbols = torch.from_numpy(y_int.copy()).float().to(device)
            z_symbols = torch.from_numpy(z_int.copy()).float().to(device)
            
            y_step_t = torch.from_numpy(y_step_arr.copy()).to(device).view(1, -1, 1, 1)
            z_step_t = torch.from_numpy(z_step_arr.copy()).to(device).view(1, -1, 1, 1)
            
            # Correct reconstruction: float_latent = integer_symbol * step_size
            y_hat_final = y_symbols * y_step_t
            z_hat_final = z_symbols * z_step_t
            
            return y_hat_final, z_hat_final, y_step_t, z_step_t
