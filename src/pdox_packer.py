import numpy as np
import zlib
import struct
import torch

class PdoxPacker:
    """
    Sovereign Bit-Packing Protocol for Paradox (.pdox) files.
    Reduces a float32 latent tensor into an ultra-compact, zlib-compressed int8 bitstream.
    """
    MAGIC = b'PDOX'

    @staticmethod
    def pack(z_q: torch.Tensor, filepath: str) -> int:
        """
        Packs a float32 latent tensor [-1.0, 1.0] into a compressed .pdox binary file.
        Returns the final size of the file in bytes.
        """
        # 1. Convert PyTorch tensor to NumPy
        z_np = z_q.detach().cpu().numpy()
        shape = z_np.shape # (B, C, H, W)
        
        # 2. Scale and Quantize float32 [-1.0, 1.0] to signed int8 [-128, 127]
        z_int8 = np.clip(np.round(z_np * 127.5), -128, 127).astype(np.int8)
        raw_bytes = z_int8.tobytes()
        
        # 3. Apply high-efficiency entropy coding (zlib level 9)
        compressed_bytes = zlib.compress(raw_bytes, level=9)
        
        # 4. Create Header: MAGIC (4 bytes) + Shape (B, C, H, W as uint16 = 8 bytes)
        header = struct.pack('>4sHHHH', PdoxPacker.MAGIC, shape[0], shape[1], shape[2], shape[3])
        
        # 5. Write to File
        with open(filepath, 'wb') as f:
            f.write(header)
            f.write(compressed_bytes)
            
        return len(header) + len(compressed_bytes)

    @staticmethod
    def unpack(filepath: str, device: str = 'cpu') -> torch.Tensor:
        """
        Reads a .pdox binary file, decompresses, and scales it back to float32.
        """
        with open(filepath, 'rb') as f:
            # 1. Unpack and validate header
            header_data = f.read(12) # 4 bytes MAGIC + 8 bytes Shape
            magic, B, C, H, W = struct.unpack('>4sHHHH', header_data)
            
            if magic != PdoxPacker.MAGIC:
                raise ValueError("Invalid file structure: Missing 'PDOX' magic token.")
                
            # 2. Read and decompress payload
            compressed_bytes = f.read()
            raw_bytes = zlib.decompress(compressed_bytes)
            
        # 3. Reconstruct the int8 array
        z_int8 = np.frombuffer(raw_bytes, dtype=np.int8).reshape(B, C, H, W)
        
        # 4. Convert back to float32 in range [-1.0, 1.0]
        z_float = z_int8.astype(np.float32) / 127.5
        
        return torch.from_numpy(z_float).to(device)
