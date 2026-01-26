"""
Loss functions for chess model training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyLoss(nn.Module):
    """
    Policy loss for move prediction.
    
    Supports both hard labels (cross-entropy) and soft labels (KL divergence).
    """
    
    def __init__(self, use_soft_labels: bool = True, label_smoothing: float = 0.0):
        """
        Initialize policy loss.
        
        Args:
            use_soft_labels: If True, use KL divergence for soft targets.
            label_smoothing: Label smoothing factor for cross-entropy.
        """
        super().__init__()
        self.use_soft_labels = use_soft_labels
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        policy_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy loss.
        
        Args:
            policy_logits: Model output logits of shape (B, NUM_MOVES).
            target: Target distribution or indices of shape (B, NUM_MOVES) or (B,).
        
        Returns:
            Scalar loss tensor.
        """
        if self.use_soft_labels:
            # KL divergence for soft targets
            log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # Add small epsilon for numerical stability
            target = target.clamp(min=1e-8)
            target = target / target.sum(dim=-1, keepdim=True)
            
            loss = F.kl_div(log_probs, target, reduction='batchmean')
        else:
            # Cross-entropy for hard labels
            if target.dim() > 1:
                # Convert soft to hard labels (argmax)
                target = target.argmax(dim=-1)
            
            if self.label_smoothing > 0:
                loss = F.cross_entropy(
                    policy_logits, target, 
                    label_smoothing=self.label_smoothing
                )
            else:
                loss = F.cross_entropy(policy_logits, target)
        
        return loss


class ValueLoss(nn.Module):
    """
    Value loss for board evaluation.
    
    Uses MSE between predicted value and target.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        value: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value loss.
        
        Args:
            value: Predicted value of shape (B, 1).
            target: Target value of shape (B, 1).
        
        Returns:
            Scalar loss tensor.
        """
        return F.mse_loss(value, target)


class DualLoss(nn.Module):
    """
    Combined policy and value loss for dual-headed models.
    """
    
    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        use_soft_labels: bool = True,
    ):
        """
        Initialize dual loss.
        
        Args:
            policy_weight: Weight for policy loss.
            value_weight: Weight for value loss.
            use_soft_labels: Use soft labels for policy.
        """
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        
        self.policy_loss = PolicyLoss(use_soft_labels=use_soft_labels)
        self.value_loss = ValueLoss()
    
    def forward(
        self,
        output: dict,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            output: Model output dict with 'policy' and 'value'.
            policy_target: Policy target.
            value_target: Value target.
        
        Returns:
            Dict with 'loss', 'policy_loss', 'value_loss'.
        """
        p_loss = self.policy_loss(output['policy'], policy_target)
        v_loss = self.value_loss(output['value'], value_target)
        
        total_loss = self.policy_weight * p_loss + self.value_weight * v_loss
        
        return {
            'loss': total_loss,
            'policy_loss': p_loss,
            'value_loss': v_loss,
        }


def create_loss(
    head_type: str,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    use_soft_labels: bool = True,
) -> nn.Module:
    """
    Factory function to create appropriate loss.
    
    Args:
        head_type: "policy", "value", or "dual".
        policy_weight: Weight for policy loss (dual only).
        value_weight: Weight for value loss (dual only).
        use_soft_labels: Use soft labels for policy.
    
    Returns:
        Loss module.
    """
    if head_type == "policy":
        return PolicyLoss(use_soft_labels=use_soft_labels)
    elif head_type == "value":
        return ValueLoss()
    elif head_type == "dual":
        return DualLoss(policy_weight, value_weight, use_soft_labels)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
