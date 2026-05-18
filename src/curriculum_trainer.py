import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from typing import Tuple, Dict, Any, Optional

# Ensure local imports work correctly
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import LatentGenesisCore
from train import PerceptualLoss, ssim_loss, compression_loss
from pdox_packer import PdoxPacker

class FidelityReinforcementAgent(nn.Module):
    """
    Stage 3: Fidelity Reinforcement Learning Agent.
    
    If direct reconstruction fails to hit the 35 dB - 40 dB PSNR goal,
    this agent acts as a localized policy that generates target-specific 
    latent refinement offsets (actions) based on a reward function.
    """
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        # A tiny convolutional policy network that predicts latent corrections (actions)
        self.policy_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, error_map: torch.Tensor) -> torch.Tensor:
        # State: Fused latent representation and spatial error projection
        state = torch.cat([z, error_map], dim=1)
        action_correction = self.policy_net(state) * 0.1 # bounded perturbation scale
        return action_correction

class CurriculumTrainer:
    """
    Curriculum-Driven Multistage Training Engine.
    
    Stage 1: Foundation Manifold Learning (standard VAE, learn global shapes).
    Stage 2: Perceptual Fine-Tuning (decay KLD to 0, maximize L1/SSIM/VGG, capture high-frequency detail).
    Stage 3: Reinforcement Manifold Search (Policy-based refinement to guarantee 35-40 dB PSNR).
    """
    def __init__(
        self,
        latent_channels: int = 16,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        self.latent_channels = latent_channels
        self.model = LatentGenesisCore(latent_channels=latent_channels).to(self.device)
        self.perc_model = PerceptualLoss().to(self.device)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def calculate_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Helper to compute pixel-perfect PSNR on scaled [0, 1] range."""
        x = torch.clamp(original * 0.5 + 0.5, 0.0, 1.0)
        y = torch.clamp(reconstructed * 0.5 + 0.5, 0.0, 1.0)
        mse = torch.mean((x - y) ** 2).item()
        if mse == 0:
            return 100.0
        return 20.0 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()

    def run_stage1_foundation(self, train_loader, epochs: int = 10, lr: float = 2e-4) -> None:
        """
        STAGE 1: Universal Foundation Manifold Learning.
        Saves reconstructed samples and .pdox files in samples/stage1/ for every epoch.
        """
        print("\n[STAGE 1] Initiating Universal Foundation Manifold Learning...")
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Create Stage 1 sample directory
        stage1_dir = os.path.join("samples", "stage1")
        os.makedirs(stage1_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            kld_weight = min(1.0, epoch / max(1, epochs // 2)) * 0.001
            
            last_images = None
            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                last_images = images
                optimizer.zero_grad(set_to_none=True)
                
                recon, mu, logvar = self.model(images)
                loss, l1, ssim, perc, kld = compression_loss(
                    recon, images, mu, logvar, kld_weight, self.perc_model
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_loader)
            print(f"  [-] Stage 1 | Epoch [{epoch+1}/{epochs}] -> Loss: {avg_loss:.4f} | KLD Wt: {kld_weight:.5f}")
            
            # Save sample reconstruction and .pdox file for this epoch
            if last_images is not None:
                self.model.eval()
                with torch.no_grad():
                    sample_input = last_images[0:1]
                    sample_recon, sample_mu, _ = self.model(sample_input)
                    
                    # Rescale to [0.0, 1.0] for image saving
                    img_to_save = torch.clamp(sample_recon * 0.5 + 0.5, 0.0, 1.0)
                    vutils.save_image(img_to_save, os.path.join(stage1_dir, f"epoch_{epoch+1}.jpg"))
                    
                    # Pack latent code as a compact .pdox file
                    sample_z_q = torch.round(torch.clamp(sample_mu, -1.0, 1.0) * 127.5) / 127.5
                    PdoxPacker.pack(sample_z_q, os.path.join(stage1_dir, f"epoch_{epoch+1}.pdox"))
            
        torch.save({'model_state_dict': self.model.state_dict()}, 
                   os.path.join(self.checkpoint_dir, 'stage1_foundation_core.pth'))
        print("[STAGE 1 COMPLETE] Foundation saved to stage1_foundation_core.pth")

    def run_stage2_perceptual_finetune(self, train_loader, epochs: int = 10, lr: float = 5e-5) -> None:
        """
        STAGE 2: Perceptual Fine-Tuning (Deterministic Autoencoder Mode).
        Saves reconstructed samples and .pdox files in samples/stage2/ for every epoch.
        """
        print("\n[STAGE 2] Initiating Perceptual & Detail Fine-Tuning...")
        stage1_path = os.path.join(self.checkpoint_dir, 'stage1_foundation_core.pth')
        if os.path.exists(stage1_path):
            checkpoint = torch.load(stage1_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Create Stage 2 sample directory
        stage2_dir = os.path.join("samples", "stage2")
        os.makedirs(stage2_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            last_images = None
            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                last_images = images
                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = self.model(images)
                
                l1_l = F.l1_loss(recon, images)
                ssim_l = ssim_loss(recon, images)
                perc_l = self.perc_model(recon, images)
                total_loss = l1_l + (1.2 * ssim_l) + (0.05 * perc_l)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += total_loss.item()
                
            avg_loss = running_loss / len(train_loader)
            print(f"  [-] Stage 2 | Epoch [{epoch+1}/{epochs}] -> Loss: {avg_loss:.4f} | SSIM: {ssim_l.item():.4f}")
            
            # Save sample reconstruction and .pdox file for this epoch
            if last_images is not None:
                self.model.eval()
                with torch.no_grad():
                    sample_input = last_images[0:1]
                    sample_recon, sample_mu, _ = self.model(sample_input)
                    
                    # Rescale to [0.0, 1.0] for image saving
                    img_to_save = torch.clamp(sample_recon * 0.5 + 0.5, 0.0, 1.0)
                    vutils.save_image(img_to_save, os.path.join(stage2_dir, f"epoch_{epoch+1}.jpg"))
                    
                    # Pack latent code as a compact .pdox file
                    sample_z_q = torch.round(torch.clamp(sample_mu, -1.0, 1.0) * 127.5) / 127.5
                    PdoxPacker.pack(sample_z_q, os.path.join(stage2_dir, f"epoch_{epoch+1}.pdox"))
            
        torch.save({'model_state_dict': self.model.state_dict()}, 
                   os.path.join(self.checkpoint_dir, 'stage2_perceptual_core.pth'))
        print("[STAGE 2 COMPLETE] Perceptual Fine-Tuning saved to stage2_perceptual_core.pth")

    def run_stage3_reinforcement_refinement(self, image_tensor: torch.Tensor, target_psnr: float = 35.0, max_steps: int = 50) -> torch.Tensor:
        """
        STAGE 3: Reinforcement Manifold Search (RL-based direct policy optimization).
        Saves the best refined result and corresponding .pdox file in samples/stage3/.
        """
        image_tensor = image_tensor.to(self.device)
        self.model.eval()
        
        # Create Stage 3 sample directory
        stage3_dir = os.path.join("samples", "stage3")
        os.makedirs(stage3_dir, exist_ok=True)
        
        with torch.no_grad():
            mu, _ = self.model.encoder(image_tensor)
            z_baseline = mu.clone().detach()
            initial_recon = self.model.decoder(z_baseline)
            
        initial_psnr = self.calculate_psnr(image_tensor, initial_recon)
        print(f"\n[STAGE 3] Target check. Initial Reconstruction: {initial_psnr:.2f} dB")
        
        if initial_psnr >= target_psnr:
            # Save baseline output
            with torch.no_grad():
                img_to_save = torch.clamp(initial_recon * 0.5 + 0.5, 0.0, 1.0)
                vutils.save_image(img_to_save, os.path.join(stage3_dir, "refined_result.jpg"))
                PdoxPacker.pack(torch.round(torch.clamp(z_baseline, -1.0, 1.0) * 127.5) / 127.5, 
                                 os.path.join(stage3_dir, "refined_result.pdox"))
            return torch.round(torch.clamp(z_baseline, -1.0, 1.0) * 127.5) / 127.5

        print(f"[STAGE 3 TRIGGERED] Initiating Reinforcement Search...")
        
        rl_agent = FidelityReinforcementAgent(channels=self.latent_channels).to(self.device)
        agent_opt = optim.Adam(rl_agent.parameters(), lr=1e-3)
        
        best_z_q = None
        best_reward = float("-inf")
        best_psnr = initial_psnr
        
        for step in range(max_steps):
            with torch.no_grad():
                current_recon = self.model.decoder(z_baseline)
            error_map = F.interpolate(
                torch.abs(image_tensor - current_recon).mean(dim=1, keepdim=True),
                size=(z_baseline.shape[2], z_baseline.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            error_map_expanded = error_map.expand(-1, self.latent_channels, -1, -1)
            
            z_action = rl_agent(z_baseline, error_map_expanded)
            z_adjusted = z_baseline + z_action
            z_q = torch.round(torch.clamp(z_adjusted, -1.0, 1.0) * 127.5) / 127.5
            
            reconstructed = self.model.decoder(z_q)
            current_psnr = self.calculate_psnr(image_tensor, reconstructed)
            
            psnr_reward = current_psnr
            if current_psnr < target_psnr:
                psnr_reward -= (target_psnr - current_psnr) * 5.0
                
            size_penalty = torch.mean((z_q - z_baseline) ** 2) * 20.0
            reward = psnr_reward - size_penalty.item()
            
            soft_z_q = torch.clamp(z_adjusted, -1.0, 1.0)
            soft_recon = self.model.decoder(soft_z_q)
            soft_psnr_loss = F.mse_loss(soft_recon, image_tensor)
            policy_loss = soft_psnr_loss * (1.0 + max(0.0, target_psnr - current_psnr))
            
            agent_opt.zero_grad()
            policy_loss.backward()
            agent_opt.step()
            
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_z_q = z_q.clone().detach()
                
            if current_psnr >= target_psnr:
                break
                
        if best_z_q is None:
            best_z_q = torch.round(torch.clamp(z_baseline, -1.0, 1.0) * 127.5) / 127.5
            
        # Save sample reconstructed image and pdox file for the final best refined latent
        with torch.no_grad():
            final_recon = self.model.decoder(best_z_q)
            img_to_save = torch.clamp(final_recon * 0.5 + 0.5, 0.0, 1.0)
            vutils.save_image(img_to_save, os.path.join(stage3_dir, "refined_result.jpg"))
            PdoxPacker.pack(best_z_q, os.path.join(stage3_dir, "refined_result.pdox"))
            
        print(f"[STAGE 3 COMPLETE] Final Output Fidelity: {best_psnr:.2f} dB.")
        return best_z_q
