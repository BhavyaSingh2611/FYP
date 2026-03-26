"""
Standard Convolutional Neural Network for chess.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ChessModel


class ConvBlock(nn.Module):
    """Single convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ConvNet(ChessModel):
    """
    Standard 6-layer ConvNet for processing 8x8 chess board tensors.

    Architecture:
        - 6 convolutional blocks with increasing channels
        - Global average pooling
        - Output dimension: channels (default 256)
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 256,
        num_layers: int = 6,
    ):
        """
        Initialize ConvNet.

        Args:
            input_channels: Number of input channels (default 18 from CNNEncoder).
            channels: Number of channels in conv layers.
            num_layers: Number of convolutional layers.
        """
        super().__init__()

        self.input_channels = input_channels
        self.channels = channels
        self.num_layers = num_layers

        # Build convolutional layers
        layers = []

        # First layer: input_channels -> channels
        layers.append(ConvBlock(input_channels, channels))

        # Remaining layers: channels -> channels
        for _ in range(num_layers - 1):
            layers.append(ConvBlock(channels, channels))

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling (handled in forward)
        self.output_dim = channels

    @property
    def name(self) -> str:
        return f"ConvNet_{self.num_layers}L_{self.channels}C"

    def get_backbone_output_dim(self) -> int:
        return self.output_dim

    def forward_backbone(self, x: torch.Tensor | dict) -> torch.Tensor:  # type: ignore[override]
        assert isinstance(x, torch.Tensor), "ConvNet expects a Tensor input"
        out = self.conv_layers(x)  # (B, C, 8, 8)

        self._spatial_features = out

        return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)  # (B, C)
