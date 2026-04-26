import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) using pre-trained AlexNet.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained AlexNet features
        self.net = alexnet(weights=AlexNet_Weights.DEFAULT).features
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
            
        # Extract features at specific layers
        self.layers = [1, 4, 7, 9, 11]
        
        # Scaling layers for LPIPS (simplified, fixed weights or learned 1x1 convs)
        # Here we use a simplified version: MSE of normalized features
        
    def forward(self, x, y):
        # Normalize inputs to expected ImageNet range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        # Assume x, y are in [0, 1] range initially or [-1, 1]
        # Let's assume input is [-1, 1] because of Tanh in decoder
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0.0
        x_feat = x
        y_feat = y
        
        layer_idx = 0
        for i, layer in enumerate(self.net):
            x_feat = layer(x_feat)
            y_feat = layer(y_feat)
            
            if i in self.layers:
                # Normalize across channel dimension
                x_norm = torch.nn.functional.normalize(x_feat, p=2, dim=1)
                y_norm = torch.nn.functional.normalize(y_feat, p=2, dim=1)
                
                # Spatial average of squared differences
                diff = (x_norm - y_norm) ** 2
                loss += diff.mean()
                layer_idx += 1
                
        return loss
