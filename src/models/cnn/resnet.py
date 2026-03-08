"""
ResNet-style Convolutional Neural Network for chess.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ChessModel


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """Residual block with skip connection and SE attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += residual
        out = F.silu(out)

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
        x = F.silu(self.initial_bn(self.initial_conv(x)))
        x = self.res_blocks(x)  # (B, C, 8, 8)

        self._spatial_features = x

        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # (B, C)
        return x
