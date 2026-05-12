import torch
import numpy as np
import zlib
import struct
import os

class HybridEntropyCoder:
    """
    Elite Hybrid Entropy Coder: The Gateway to the .bpox (Blueprint) Binary Format.
    
    Handles both Polynomial coefficients (Rule-based) and Neural latents.
    
    Format Structure:
    - Header: Magic(4), Version(4), Rows(4), Cols(4), DecisionMap(4)
    - Payload: Interleaved Math Coeffs and Neural Blobs
    """
    def __init__(self, magic_bytes=b"BPOX", version=7):
        self.magic_bytes = magic_bytes
        self.version = version

    def compress(self, encoding, output_path):
        """
        Compresses the hybrid encoding dictionary into a binary file.
        
        encoding = {
            'grid': (rows, cols),
            'decisions': (B, num_tiles),
            'poly_coeffs': {idx: coeffs},
            'neural_latents': {idx: {y_hat, y_step, z_hat, z_step, orig_h, orig_w}}
        }
        """
        grid = encoding['grid']
        decisions = encoding['decisions'][0].cpu().numpy() # Assume batch size 1 for inference
        num_tiles = len(decisions)
        
        # 1. Prepare Header
        # Decision Map packed into a bit-field (supporting up to 32 tiles in 4 bytes)
        decision_map = 0
        for i, d in enumerate(decisions):
            if d > 0:
                decision_map |= (1 << i)
        
        header = struct.pack(">4sIIIII", 
                             self.magic_bytes, self.version,
                             grid[0], grid[1], num_tiles, decision_map)
        
        payload = bytearray()
        
        # 2. Pack Tiles
        for i in range(num_tiles):
            if decisions[i] == 0:
                # Math Tile: Pack 18 coefficients (3 channels * 6 coeffs)
                coeffs = encoding['poly_coeffs'][i][0].cpu().numpy().astype(np.float32) # (3, 6)
                payload.extend(coeffs.tobytes())
            else:
                # Neural Tile: Use the Aether binary logic
                lat = encoding['neural_latents'][i]
                
                # Convert latents to symbols
                y_step = lat['y_step'].detach().cpu().view(1, -1, 1, 1).float()
                y_symbols = torch.round(lat['y_hat'].detach().cpu().float() / torch.clamp(y_step, min=1e-6))
                y_int = y_symbols.numpy().astype(np.int16)
                
                z_int = None
                z_step_np = None
                if lat.get('z_hat') is not None:
                    z_step = lat['z_step'].detach().cpu().view(1, -1, 1, 1).float()
                    z_symbols = torch.round(lat['z_hat'].detach().cpu().float() / torch.clamp(z_step, min=1e-6))
                    z_int = z_symbols.numpy().astype(np.int16)
                    z_step_np = lat['z_step'].detach().cpu().view(-1).numpy().astype(np.float32)
                
                y_step_np = lat['y_step'].detach().cpu().view(-1).numpy().astype(np.float32)
                
                # Compress payloads
                y_blob = zlib.compress(y_int.tobytes(), level=9)
                z_blob = zlib.compress(z_int.tobytes(), level=9) if z_int is not None else b""
                
                # Pack neural header for this tile
                # Shapes: b, cy, hy, wy, cz, hz, wz, orig_h, orig_w, len_y, len_z
                b, cy, hy, wy = y_int.shape
                cz, hz, wz = (0, 0, 0)
                if z_int is not None:
                    _, cz, hz, wz = z_int.shape
                
                neural_header = struct.pack(">IIIIIIIII II", 
                                          b, cy, hy, wy, cz, hz, wz, 
                                          lat['orig_h'], lat['orig_w'],
                                          len(y_blob), len(z_blob))
                
                payload.extend(neural_header)
                payload.extend(y_step_np.tobytes())
                if z_step_np is not None:
                    payload.extend(z_step_np.tobytes())
                payload.extend(y_blob)
                payload.extend(z_blob)
        
        # 3. Write File
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(payload)
            
        return os.path.getsize(output_path)

    def decompress(self, input_path, device='cuda'):
        """
        Reconstructs the hybrid encoding from a binary file.
        """
        with open(input_path, 'rb') as f:
            # 1. Read Header
            header_size = struct.calcsize(">4sIIIII")
            header_data = f.read(header_size)
            magic, ver, rows, cols, num_tiles, decision_map = struct.unpack(">4sIIIII", header_data)
            
            if magic != self.magic_bytes:
                raise ValueError("Invalid Blueprint file signature")
            
            decisions = []
            for i in range(num_tiles):
                decisions.append(1 if (decision_map & (1 << i)) else 0)
            
            decisions_t = torch.tensor([decisions], device=device)
            
            poly_coeffs = {}
            neural_latents = {}
            
            # 2. Unpack Tiles
            for i in range(num_tiles):
                if decisions[i] == 0:
                    # Math Tile: 18 floats
                    coeff_data = f.read(18 * 4)
                    coeffs = np.frombuffer(coeff_data, dtype=np.float32).reshape(1, 3, 6)
                    poly_coeffs[i] = torch.from_numpy(coeffs).to(device)
                else:
                    # Neural Tile
                    n_header_size = struct.calcsize(">IIIIIIIII II")
                    n_header_data = f.read(n_header_size)
                    b, cy, hy, wy, cz, hz, wz, orig_h, orig_w, len_y, len_z = struct.unpack(">IIIIIIIII II", n_header_data)
                    
                    y_step_data = f.read(cy * 4)
                    y_step_arr = np.frombuffer(y_step_data, dtype=np.float32)
                    
                    z_step_arr = None
                    if cz > 0:
                        z_step_data = f.read(cz * 4)
                        z_step_arr = np.frombuffer(z_step_data, dtype=np.float32)
                    
                    y_blob = f.read(len_y)
                    z_blob = f.read(len_z)
                    
                    y_int_data = zlib.decompress(y_blob)
                    y_int = np.frombuffer(y_int_data, dtype=np.int16).reshape(b, cy, hy, wy)
                    
                    z_hat_t = None
                    z_step_t = None
                    if len_z > 0:
                        z_int_data = zlib.decompress(z_blob)
                        z_int = np.frombuffer(z_int_data, dtype=np.int16).reshape(b, cz, hz, wz)
                        z_symbols = torch.from_numpy(z_int.copy()).float().to(device)
                        z_step_t = torch.from_numpy(z_step_arr.copy()).to(device).view(1, -1, 1, 1)
                        z_hat_t = z_symbols * z_step_t
                    
                    y_symbols = torch.from_numpy(y_int.copy()).float().to(device)
                    y_step_t = torch.from_numpy(y_step_arr.copy()).to(device).view(1, -1, 1, 1)
                    y_hat_t = y_symbols * y_step_t
                    
                    neural_latents[i] = {
                        'y_hat': y_hat_t,
                        'y_step': y_step_t,
                        'z_hat': z_hat_t,
                        'z_step': z_step_t,
                        'orig_h': orig_h,
                        'orig_w': orig_w
                    }
                    
        return {
            'grid': (rows, cols),
            'decisions': decisions_t,
            'poly_coeffs': poly_coeffs,
            'neural_latents': neural_latents,
            'num_tiles': num_tiles
        }
