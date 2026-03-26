"""
Output heads for chess models: Policy, Value, and Dual-headed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chess_env.move_index import NUM_MOVES


class PolicyHead(nn.Module):
    """
    Policy head that preserves spatial information from CNN feature maps.

    Uses a 1×1 conv to reduce channels, flattens the 8×8 grid, then
    projects to the move space.  Falls back to a linear head when no
    spatial features are available (non-CNN backbones).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 256, spatial_channels: int = 32
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, spatial_channels, 1)
        self.bn_conv = nn.BatchNorm2d(spatial_channels)
        flat_dim = spatial_channels * 64  # 8×8
        self.fc = nn.Linear(flat_dim, NUM_MOVES)

        self.fc_fallback = nn.Linear(input_dim, hidden_dim)
        self.bn_fallback = nn.BatchNorm1d(hidden_dim)
        self.fc_fallback2 = nn.Linear(hidden_dim, NUM_MOVES)

    def forward(self, x: torch.Tensor, spatial: torch.Tensor | None = None) -> dict:
        if spatial is not None:
            s = F.relu(self.bn_conv(self.conv(spatial)))
            s = s.view(s.size(0), -1)
            return {"policy": self.fc(s)}

        x = F.relu(self.bn_fallback(self.fc_fallback(x)))
        return {"policy": self.fc_fallback2(x)}


class ValueHead(nn.Module):
    """
    Value head that outputs a board evaluation score.

    Output: (batch_size, 1) value in range [-1, 1]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize value head.

        Args:
            input_dim: Dimension of backbone output.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, spatial: torch.Tensor | None = None) -> dict:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        value = torch.tanh(self.fc3(x))
        return {"value": value}


class DualHead(nn.Module):
    """
    Dual head that outputs both policy and value (AlphaZero-style).
    Uses spatial features for the policy branch when available.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 256, spatial_channels: int = 32
    ):
        super().__init__()

        self.policy_conv = nn.Conv2d(input_dim, spatial_channels, 1)
        self.policy_bn_conv = nn.BatchNorm2d(spatial_channels)
        self.policy_fc_spatial = nn.Linear(spatial_channels * 64, NUM_MOVES)

        self.policy_fc1 = nn.Linear(input_dim, hidden_dim)
        self.policy_bn1 = nn.BatchNorm1d(hidden_dim)
        self.policy_fc2 = nn.Linear(hidden_dim, NUM_MOVES)

        self.value_fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_bn1 = nn.BatchNorm1d(hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.value_fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, spatial: torch.Tensor | None = None) -> dict:
        if spatial is not None:
            s = F.relu(self.policy_bn_conv(self.policy_conv(spatial)))
            s = s.view(s.size(0), -1)
            policy_logits = self.policy_fc_spatial(s)
        else:
            p = F.relu(self.policy_bn1(self.policy_fc1(x)))
            policy_logits = self.policy_fc2(p)

        v = F.relu(self.value_bn1(self.value_fc1(x)))
        v = F.relu(self.value_bn2(self.value_fc2(v)))
        value = torch.tanh(self.value_fc3(v))

        return {"policy": policy_logits, "value": value}


def create_head(head_type: str, input_dim: int, hidden_dim: int = 256) -> nn.Module:
    """
    Factory function to create a head.

    Args:
        head_type: "policy", "value", or "dual"
        input_dim: Backbone output dimension.
        hidden_dim: Hidden layer dimension.

    Returns:
        Head module.
    """
    if head_type == "policy":
        return PolicyHead(input_dim, hidden_dim)
    elif head_type == "value":
        return ValueHead(input_dim, hidden_dim)
    elif head_type == "dual":
        return DualHead(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
