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
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GNN models")
        
        # GAT with concatenated heads
        self.conv = GATConv(
            in_channels, 
            out_channels // heads,  # Each head outputs out_channels // heads
            heads=heads,
            concat=True,
            dropout=0.1,
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
    ) -> torch.Tensor:
        residual = x
        if self.residual is not None:
            residual = self.residual(residual)
        
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.elu(out + residual)
        
        return out


class GAT(ChessModel):
    """
    Graph Attention Network for chess.
    
    Architecture:
        - Initial projection from node features to hidden dim
        - N GAT layers with multi-head attention and residual connections
        - Global mean pooling over all nodes
        - Output dimension: hidden_dim
    
    GAT uses attention mechanisms to learn which edges are more important,
    making it particularly suitable for dynamic edge structures.
    
    Supports three edge types:
        - static: King-move adjacency (fixed graph)
        - dynamic: Legal moves as edges (changes each position)
        - hybrid: Both edge types (dual-stream processing)
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        edge_type: str = "hybrid",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize GAT.
        
        Args:
            input_dim: Node feature dimension (12 from GNNEncoder).
            hidden_dim: Hidden dimension (must be divisible by heads).
            num_layers: Number of GAT layers.
            edge_type: "static", "dynamic", or "hybrid".
            heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GNN models")
        
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_type = edge_type
        self.heads = heads
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers for spatial edges
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, heads=heads)
            for _ in range(num_layers)
        ])
        
        # For hybrid: separate GAT layers for legal move edges
        if edge_type == "hybrid":
            self.legal_gat_layers = nn.ModuleList([
                GATLayer(hidden_dim, hidden_dim, heads=heads)
                for _ in range(num_layers)
            ])
            # Attention-based fusion
            self.fusion_attention = nn.Linear(hidden_dim * 2, 2)
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        
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
        return f"GAT_{self.edge_type}_{self.num_layers}L_{self.heads}H_{self.hidden_dim}D"
    
    def get_backbone_output_dim(self) -> int:
        return self.output_dim
    
    def forward_backbone(self, x: dict) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Dictionary with:
                - 'x': Node features of shape (B*64, 12) or (B, 64, 12)
                - 'edge_index': Edge indices (format depends on edge_type)
                - 'side_to_move': Side to move of shape (B,)
                - 'castling': Castling rights of shape (B, 4)
        
        Returns:
            Feature tensor of shape (B, hidden_dim).
        """
        node_features = x['x']
        edge_index = x['edge_index']
        side_to_move = x['side_to_move']
        castling = x['castling']
        
        # Handle batched input
        if node_features.dim() == 3:
            batch_size = node_features.size(0)
            node_features = node_features.view(-1, node_features.size(-1))
            
            batch = torch.arange(batch_size, device=node_features.device)
            batch = batch.unsqueeze(1).expand(-1, 64).reshape(-1)
            
            if self.edge_type == "hybrid":
                edge_index = self._batch_edge_indices_hybrid(edge_index, batch_size)
            else:
                edge_index = self._batch_edge_indices(edge_index, batch_size)
        else:
            batch_size = 1
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        
        # Initial projection
        h = self.input_proj(node_features)
        h = self.dropout(F.relu(h))
        
        if self.edge_type == "hybrid":
            # Dual-stream processing with attention fusion
            h_spatial = h
            for layer in self.gat_layers:
                h_spatial = layer(h_spatial, edge_index['spatial'])
            
            h_legal = h
            for layer in self.legal_gat_layers:
                h_legal = layer(h_legal, edge_index['legal'])
            
            # Attention-based fusion
            combined = torch.cat([h_spatial, h_legal], dim=-1)
            attention_weights = F.softmax(self.fusion_attention(combined), dim=-1)
            
            h = attention_weights[:, 0:1] * h_spatial + attention_weights[:, 1:2] * h_legal
            h = self.fusion_proj(h)
        else:
            for layer in self.gat_layers:
                h = layer(h, edge_index)
        
        # Global mean pooling
        output = global_mean_pool(h, batch)
        
        # Add side to move
        side_emb = self.side_embedding(side_to_move)
        output = output + side_emb
        
        # Add castling info
        castling_emb = self.castling_proj(castling)
        output = output + castling_emb
        
        # Final projection
        output = self.output_proj(output)
        
        return output
    
    def _batch_edge_indices(
        self,
        edge_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Batch edge indices by offsetting node indices."""
        edge_indices = []
        for i in range(batch_size):
            offset = i * 64
            edge_indices.append(edge_index + offset)
        return torch.cat(edge_indices, dim=1)
    
    def _batch_edge_indices_hybrid(
        self,
        edge_index: dict,
        batch_size: int,
    ) -> dict:
        """Batch hybrid edge indices."""
        return {
            'spatial': self._batch_edge_indices(edge_index['spatial'], batch_size),
            'legal': self._batch_edge_indices(edge_index['legal'], batch_size),
        }
