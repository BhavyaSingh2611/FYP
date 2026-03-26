"""
Abstract base class for chess neural network models.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ChessModel(ABC, nn.Module):
    """
    Abstract base class for all chess neural network models.

    All models must implement:
        - forward(): Process input and return backbone features
        - get_backbone_output_dim(): Return the dimension of backbone output
    """

    def __init__(self) -> None:
        super().__init__()
        self.head: nn.Module | None = None
        self._spatial_features: torch.Tensor | None = None

    @abstractmethod
    def forward_backbone(self, x: torch.Tensor | dict) -> torch.Tensor:
        """
        Forward pass through the backbone network only.

        Args:
            x: Input tensor or dictionary (format depends on model type).

        Returns:
            Feature tensor of shape (batch_size, backbone_output_dim).
        """
        pass

    @abstractmethod
    def get_backbone_output_dim(self) -> int:
        """
        Get the output dimension of the backbone.

        Returns:
            Integer dimension of the backbone output.
        """
        pass

    def forward(self, x: torch.Tensor | dict) -> dict:
        features = self.forward_backbone(x)

        if self.head is None:
            raise RuntimeError("Model head not set. Call set_head() first.")

        spatial = getattr(self, "_spatial_features", None)
        if spatial is not None:
            self._spatial_features = None
            return dict(self.head(features, spatial=spatial))

        return dict(self.head(features))

    def set_head(self, head: nn.Module) -> None:
        """
        Set the output head for the model.

        Args:
            head: A PolicyHead, ValueHead, or DualHead module.
        """
        self.head = head

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the model name."""
        pass

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
