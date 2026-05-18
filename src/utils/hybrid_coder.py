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
    Format Structure:
    - Header: Magic(4), Version(4), Rows(4), Cols(4), DecisionMap(4), NumCoeffs(4)
    - Payload: Interleaved Math Coeffs and Neural Blobs
    """
    def __init__(self, magic_bytes=b"BPOX", version=8):
        self.magic_bytes = magic_bytes
        self.version = version

    def compress(self, encoding, output_path):
        """
        Compresses the vectorized hybrid encoding into a .bpox binary file.
        """
        grid = encoding['grid']
        decisions = encoding['decisions'][0].cpu().numpy() # (num_tiles,)
        num_tiles = len(decisions)
        
        # 1. Prepare Header
        decision_map = 0
        for i, d in enumerate(decisions):
            if d > 0:
                decision_map |= (1 << i)
        
        num_coeffs = encoding['poly_coeffs_all'].shape[3]
        
        header = struct.pack(">4sIIIIII", 
                             self.magic_bytes, self.version,
                             grid[0], grid[1], num_tiles, decision_map, num_coeffs)
        
        payload = bytearray()
        
        # 2. Extract Data from Vectorized Encodings
        poly_coeffs_all = encoding['poly_coeffs_all'][0].cpu().numpy() # (num_tiles, 3, num_coeffs)
        
        neural_data = encoding['neural_data']
        neu_idx = 0 # Pointer into neural batch
        
        # 3. Pack Tiles Interleaved
        for i in range(num_tiles):
            if decisions[i] == 0:
                # Math Tile: num_coeffs * 3 channels
                coeffs = poly_coeffs_all[i].astype(np.float32) # (3, num_coeffs)
                payload.extend(coeffs.tobytes())
            else:
                # Neural Tile: Extract from batch
                if neural_data is None:
                    continue
                
                # Slice this tile's latents from the neural batch
                y_hat = neural_data['metrics']['y_hat'][neu_idx:neu_idx+1]
                y_step = neural_data['metrics']['y_step'][neu_idx:neu_idx+1]
                
                # Convert to symbols
                y_symbols = torch.round(y_hat.detach().cpu().float() / torch.clamp(y_step.detach().cpu(), min=1e-6))
                y_int = y_symbols.numpy().astype(np.int16)
                y_step_np = y_step.detach().cpu().view(-1).numpy().astype(np.float32)
                
                # Optional hyperprior latents
                z_blob = b""
                z_step_np = None
                cz, hz, wz = (0, 0, 0)
                
                if 'z_hat' in neural_data['metrics'] and neural_data['metrics']['z_hat'] is not None:
                    z_hat = neural_data['metrics']['z_hat'][neu_idx:neu_idx+1]
                    z_step = neural_data['metrics']['z_step'][neu_idx:neu_idx+1]
                    z_symbols = torch.round(z_hat.detach().cpu().float() / torch.clamp(z_step.detach().cpu(), min=1e-6))
                    z_int = z_symbols.numpy().astype(np.int16)
                    z_step_np = z_step.detach().cpu().view(-1).numpy().astype(np.float32)
                    _, cz, hz, wz = z_int.shape
                    z_blob = zlib.compress(z_int.tobytes(), level=9)
                
                y_blob = zlib.compress(y_int.tobytes(), level=9)
                
                # Local Neural Header
                b, cy, hy, wy = y_int.shape
                # Use standard 160x160 for hybrid tiles (padded)
                neural_header = struct.pack(">IIIIIIIII II", 
                                          b, cy, hy, wy, cz, hz, wz, 
                                          160, 160,
                                          len(y_blob), len(z_blob))
                
                payload.extend(neural_header)
                payload.extend(y_step_np.tobytes())
                if z_step_np is not None:
                    payload.extend(z_step_np.tobytes())
                payload.extend(y_blob)
                payload.extend(z_blob)
                
                neu_idx += 1
        
        # 4. Write File
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(payload)
            
        return os.path.getsize(output_path)

    def decompress(self, input_path, device='cuda'):
        """
        Reconstructs the vectorized hybrid encoding from a .bpox file.
        """
        with open(input_path, 'rb') as f:
            header_size = struct.calcsize(">4sIIIIII")
            header_data = f.read(header_size)
            magic, ver, rows, cols, num_tiles, decision_map, num_coeffs = struct.unpack(">4sIIIIII", header_data)
            
            decisions = []
            for i in range(num_tiles):
                decisions.append(1 if (decision_map & (1 << i)) else 0)
            
            decisions_t = torch.tensor([decisions], device=device)
            poly_coeffs_all = torch.zeros(1, num_tiles, 3, num_coeffs, device=device)
            
            y_hat_list = []
            y_step_list = []
            z_hat_list = []
            z_step_list = []
            complex_mask = []
            
            for i in range(num_tiles):
                if decisions[i] == 0:
                    # Math: num_coeffs * 3 floats
                    coeff_data = f.read(num_coeffs * 3 * 4)
                    coeffs = np.frombuffer(coeff_data, dtype=np.float32).reshape(3, num_coeffs)
                    poly_coeffs_all[0, i] = torch.from_numpy(coeffs).to(device)
                    complex_mask.append(False)
                else:
                    # Neural
                    n_header_size = struct.calcsize(">IIIIIIIII II")
                    n_header_data = f.read(n_header_size)
                    b, cy, hy, wy, cz, hz, wz, oh, ow, len_y, len_z = struct.unpack(">IIIIIIIII II", n_header_data)
                    
                    y_step_arr = np.frombuffer(f.read(cy * 4), dtype=np.float32)
                    z_step_arr = None
                    if cz > 0:
                        z_step_arr = np.frombuffer(f.read(cz * 4), dtype=np.float32)
                    
                    y_blob = f.read(len_y)
                    z_blob = f.read(len_z)
                    
                    y_int = np.frombuffer(zlib.decompress(y_blob), dtype=np.int16).reshape(b, cy, hy, wy)
                    y_hat_t = torch.from_numpy(y_int.copy()).float().to(device) * torch.from_numpy(y_step_arr).to(device).view(1, -1, 1, 1)
                    y_hat_list.append(y_hat_t)
                    y_step_list.append(torch.from_numpy(y_step_arr).to(device).view(1, -1, 1, 1))
                    
                    if len_z > 0:
                        z_int = np.frombuffer(zlib.decompress(z_blob), dtype=np.int16).reshape(b, cz, hz, wz)
                        z_hat_t = torch.from_numpy(z_int.copy()).float().to(device) * torch.from_numpy(z_step_arr).to(device).view(1, -1, 1, 1)
                        z_hat_list.append(z_hat_t)
                        z_step_list.append(torch.from_numpy(z_step_arr).to(device).view(1, -1, 1, 1))
                    
                    complex_mask.append(True)
            
            neural_data = None
            if any(complex_mask):
                neural_data = {
                    'metrics': {
                        'y_hat': torch.cat(y_hat_list, dim=0),
                        'y_step': torch.cat(y_step_list, dim=0),
                        'z_hat': torch.cat(z_hat_list, dim=0) if z_hat_list else None,
                        'z_step': torch.cat(z_step_list, dim=0) if z_step_list else None,
                    },
                    'mask': torch.tensor(complex_mask, device=device)
                }
                
        return {
            'grid': (rows, cols),
            'decisions': decisions_t,
            'poly_coeffs_all': poly_coeffs_all,
            'neural_data': neural_data,
            'num_tiles': num_tiles
        }
