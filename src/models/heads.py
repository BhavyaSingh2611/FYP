"""
Output heads for chess models: Policy, Value, and Dual-headed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chess_env.board_wrapper import NUM_MOVES


class PolicyHead(nn.Module):
    """
    Policy head that outputs a probability distribution over all moves.
    
    Output: (batch_size, NUM_MOVES) logits
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize policy head.
        
        Args:
            input_dim: Dimension of backbone output.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, NUM_MOVES)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Backbone features of shape (batch_size, input_dim).
        
        Returns:
            Dictionary with 'policy' logits of shape (batch_size, NUM_MOVES).
        """
        x = F.relu(self.bn1(self.fc1(x)))
        policy_logits = self.fc2(x)
        
        return {'policy': policy_logits}


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
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Backbone features of shape (batch_size, input_dim).
        
        Returns:
            Dictionary with 'value' of shape (batch_size, 1) in range [-1, 1].
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        value = torch.tanh(self.fc3(x))
        
        return {'value': value}


class DualHead(nn.Module):
    """
    Dual head that outputs both policy and value (AlphaZero-style).
    
    Outputs:
        - policy: (batch_size, NUM_MOVES) logits
        - value: (batch_size, 1) value in range [-1, 1]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize dual head.
        
        Args:
            input_dim: Dimension of backbone output.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        
        # Policy branch
        self.policy_fc1 = nn.Linear(input_dim, hidden_dim)
        self.policy_bn1 = nn.BatchNorm1d(hidden_dim)
        self.policy_fc2 = nn.Linear(hidden_dim, NUM_MOVES)
        
        # Value branch
        self.value_fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_bn1 = nn.BatchNorm1d(hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.value_fc3 = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Backbone features of shape (batch_size, input_dim).
        
        Returns:
            Dictionary with:
                - 'policy': logits of shape (batch_size, NUM_MOVES)
                - 'value': value of shape (batch_size, 1) in range [-1, 1]
        """
        # Policy branch
        p = F.relu(self.policy_bn1(self.policy_fc1(x)))
        policy_logits = self.policy_fc2(p)
        
        # Value branch
        v = F.relu(self.value_bn1(self.value_fc1(x)))
        v = F.relu(self.value_bn2(self.value_fc2(v)))
        value = torch.tanh(self.value_fc3(v))
        
        return {'policy': policy_logits, 'value': value}


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
