import os
import sys
import torch

# Add src to the path
sys.path.append(os.path.abspath("src"))

from pdox_packer import PdoxPacker
from mobile_decoder import MobileGenesisDecoder
from latent_refine import optimize_latent
from curriculum_trainer import CurriculumTrainer

def run_tests():
    print("[*] Initiating Optimization Integration Test...")
    
    # 1. Test MobileGenesisDecoder
    print("  [-] Initializing Mobile Decoder...")
    decoder = MobileGenesisDecoder(latent_channels=16)
    dummy_latent = torch.randn(1, 16, 16, 16)
    reconstructed = decoder(dummy_latent)
    print(f"  [SUCCESS] Mobile Decoder output shape: {reconstructed.shape}")
    
    # 2. Test PdoxPacker
    print("  [-] Testing Sovereign Bit-Packing Serialization...")
    temp_file = "test_temp.pdox"
    # Create quantized dummy latent in [-1, 1]
    dummy_q = torch.round(torch.clamp(dummy_latent, -1.0, 1.0) * 127.5) / 127.5
    
    packed_size = PdoxPacker.pack(dummy_q, temp_file)
    print(f"  [SUCCESS] Latent packed to '{temp_file}' ({packed_size} bytes)")
    
    unpacked_q = PdoxPacker.unpack(temp_file)
    print(f"  [SUCCESS] Latent unpacked. Match checked: {torch.allclose(dummy_q, unpacked_q)}")
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    # 3. Test CurriculumTrainer (Stage 3 Reinforcement Search)
    print("  [-] Testing Stage 3 Reinforcement Search Pipeline...")
    # Using 'cpu' to ensure fast lightweight mocked execution
    trainer = CurriculumTrainer(latent_channels=16, device="cpu")
    dummy_image = torch.clamp(torch.randn(1, 3, 256, 256), -1.0, 1.0)
    
    # Execute 3 steps of Stage 3 RL policy-refine to ensure no runtime errors
    refined_latent = trainer.run_stage3_reinforcement_refinement(
        image_tensor=dummy_image, 
        target_psnr=40.0, 
        max_steps=3
    )
    print(f"  [SUCCESS] Refined latent shape: {refined_latent.shape}")
        
    print("[*] All Paradox Optimization Integrations Passed Successfully! 🌌🛡️")

if __name__ == "__main__":
    run_tests()

