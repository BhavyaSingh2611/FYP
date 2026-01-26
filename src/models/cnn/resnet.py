"""
ResNet-style Convolutional Neural Network for chess.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ChessModel


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class ResNet(ChessModel):
    """
    ResNet-style CNN with skip connections for chess.
    
    Architecture:
        - Initial conv layer to project to channels
        - N residual blocks
        - Global average pooling
        - Output dimension: channels (default 256)
    
    Configurable depths: 6, 10, 20 blocks.
    """
    
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 256,
        num_blocks: int = 10,
    ):
        """
        Initialize ResNet.
        
        Args:
            input_channels: Number of input channels (default 18).
            channels: Number of channels in residual blocks.
            num_blocks: Number of residual blocks (6, 10, or 20).
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.channels = channels
        self.num_blocks = num_blocks
        
        # Initial projection
        self.initial_conv = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        
        self.output_dim = channels
    
    @property
    def name(self) -> str:
        return f"ResNet_{self.num_blocks}B_{self.channels}C"
    
    def get_backbone_output_dim(self) -> int:
        return self.output_dim
    
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, 8, 8).
        
        Returns:
            Feature tensor of shape (batch_size, channels).
        """
        # Initial projection
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        x = self.res_blocks(x)  # (B, C, 8, 8)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        
        return x
