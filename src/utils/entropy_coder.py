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
    """
    def __init__(self, magic_bytes=b"PDOX", version=5):
        self.magic_bytes = magic_bytes
        self.version = version

    def compress(self, y_hat, z_hat, y_step, z_step, output_path):
        """
        Compresses the neural latents and their quantization metadata.
        
        y_hat: Quantized main latent (B, C, H, W)
        z_hat: Quantized hyper-latent (B, C, H, W)
        y_step: Per-channel quantization step for y
        z_step: Per-channel quantization step for z
        """
        # 1. Convert to Int16 (Arithmetic coding usually works on integers)
        # Note: We divide by step inside the quantizer, so y_hat is already an integer representation
        y_int = y_hat.detach().cpu().numpy().astype(np.int16)
        z_int = z_hat.detach().cpu().numpy().astype(np.int16)
        
        # 2. Metadata preservation (CRITICAL for reconstruction)
        y_step_arr = y_step.detach().cpu().numpy().astype(np.float32)
        z_step_arr = z_step.detach().cpu().numpy().astype(np.float32)
        
        # 3. Serialize shapes
        b, cy, hy, wy = y_int.shape
        _, cz, hz, wz = z_int.shape
        
        # 4. Pack Payloads
        # In a production environment, this is where torchac (Arithmetic Coding) happens.
        # Here we use zlib as a high-performance bitstream simulator.
        y_payload = zlib.compress(y_int.tobytes(), level=9)
        z_payload = zlib.compress(z_int.tobytes(), level=9)
        
        # 5. Build Header (Magic, Ver, B, CY, HY, WY, CZ, HZ, WZ, LenY, LenZ, Steps...)
        # We store y_step and z_step (each 192 and 128 floats respectively)
        header_base = struct.pack(">4sIIIIIIIIII", 
                             self.magic_bytes, self.version,
                             b, cy, hy, wy, cz, hz, wz,
                             len(y_payload), len(z_payload))
        
        y_step_data = y_step_arr.tobytes()
        z_step_data = z_step_arr.tobytes()
        
        # 6. Write Atomic .padox File
        with open(output_path, 'wb') as f:
            f.write(header_base)
            f.write(y_step_data) # [192 * 4 bytes]
            f.write(z_step_data) # [128 * 4 bytes]
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
            # Shapes: cy for y, cz for z
            y_step_data = f.read(cy * 4)
            z_step_data = f.read(cz * 4)
            
            y_step_arr = np.frombuffer(y_step_data, dtype=np.float32)
            z_step_arr = np.frombuffer(z_step_data, dtype=np.float32)
            
            # 3. Read and Decompress Payloads
            y_payload = f.read(len_y)
            z_payload = f.read(len_z)
            
            y_int_data = zlib.decompress(y_payload)
            z_int_data = zlib.decompress(z_payload)
            
            # 4. Reconstruct Tensors
            y_int = np.frombuffer(y_int_data, dtype=np.int16).reshape(b, cy, hy, wy)
            z_int = np.frombuffer(z_int_data, dtype=np.int16).reshape(b, cz, hz, wz)
            
            # 5. RESCALE (The 'Elite' part)
            # We must turn integers back into the correct latent values using the steps
            y_hat = torch.from_numpy(y_int.copy()).float().to(device)
            z_hat = torch.from_numpy(z_int.copy()).float().to(device)
            
            y_step_t = torch.from_numpy(y_step_arr).to(device).view(1, -1, 1, 1)
            z_step_t = torch.from_numpy(z_step_arr).to(device).view(1, -1, 1, 1)
            
            # Correct mathematical reconstruction: y_hat_rescaled = y_int * step
            # Actually, y_hat from decompress is already the 'integer' part, we must rescale.
            y_hat_final = y_hat * y_step_t
            z_hat_final = z_hat * z_step_t
            
            return y_hat_final, z_hat_final, y_step_t, z_step_t
