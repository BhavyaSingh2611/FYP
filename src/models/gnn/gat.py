"""
Graph Attention Network (GAT) for chess.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ChessModel

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class GATLayer(nn.Module):
    """Single GAT layer with multi-head attention and residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, edge_dim: int = 3):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GNN models")
        
        # GAT with concatenated heads and edge features
        self.conv = GATConv(
            in_channels, 
            out_channels // heads,  # Each head outputs out_channels // heads
            heads=heads,
            concat=True,
            dropout=0.1,
            edge_dim=edge_dim
        )
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Residual projection if dimensions don't match
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = x
        if self.residual is not None:
            residual = self.residual(residual)
        
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        out = self.bn(out)
        out = F.elu(out + residual)
        
        return out


class GAT(ChessModel):
    """
    Graph Attention Network for chess with Edge Features.
    
    Architecture:
        - Initial projection
        - N GAT layers with edge attributes
        - Global mean pooling
        - Output dimension: hidden_dim
    """
    
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 256,
        num_layers: int = 5,
        edge_type: str = "hybrid", # Kept for config compat, effectively unused
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GNN models")
        
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, heads=heads, edge_dim=3)
            for _ in range(num_layers)
        ])
        
        # Side to move embedding
        self.side_embedding = nn.Embedding(2, hidden_dim)
        
        # Castling projection
        self.castling_proj = nn.Linear(4, hidden_dim)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim
    
    @property
    def name(self) -> str:
        return f"GAT_{self.num_layers}L_{self.heads}H_{self.hidden_dim}D"
    
    def get_backbone_output_dim(self) -> int:
        return self.output_dim
    
    def forward_backbone(self, x: dict) -> torch.Tensor:
        node_features = x['x']
        edge_index = x['edge_index']
        edge_attr = x.get('edge_attr')
        side_to_move = x['side_to_move']
        castling = x['castling']

        if 'batch' in x:
            batch = x['batch'].to(node_features.device)
        elif node_features.dim() == 3:
            batch_size = node_features.size(0)
            node_features = node_features.view(-1, node_features.size(-1))
            batch = torch.arange(batch_size, device=node_features.device)
            batch = batch.unsqueeze(1).expand(-1, 64).reshape(-1)
            edge_indices, edge_attrs = [], []
            for i in range(batch_size):
                edge_indices.append(edge_index + i * 64)
                if edge_attr is not None:
                    edge_attrs.append(edge_attr)
            edge_index = torch.cat(edge_indices, dim=1)
            if edge_attrs:
                edge_attr = torch.cat(edge_attrs, dim=0)
        else:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)

        h = self.input_proj(node_features)
        h = self.dropout(F.relu(h))

        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_attr)

        output = global_mean_pool(h, batch)

        side_emb = self.side_embedding(side_to_move)
        output = output + side_emb

        castling_emb = self.castling_proj(castling)
        output = output + castling_emb

        output = self.output_proj(output)

        return output



