"""
Graph Convolutional Network (GCN) for chess.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ChessModel

try:
    from torch_geometric.nn import GCNConv, global_mean_pool

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class GCNLayer(nn.Module):
    """Single GCN layer with BatchNorm and residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GNN models")

        self.conv = GCNConv(in_channels, out_channels)
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
        out = F.relu(out + residual)

        return out


class GCN(ChessModel):
    """
    Graph Convolutional Network for chess.

    Architecture:
        - Initial projection from node features to hidden dim
        - N GCN layers with residual connections
        - Global mean pooling over all nodes
        - Output dimension: hidden_dim

    Note: GCN uses the graph structure provided by the encoder (Adjacency).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 256,
        num_layers: int = 6,
        edge_type: str = "hybrid",  # Kept for compat
        dropout: float = 0.1,
    ):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GNN models")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Side to move embedding
        self.side_embedding = nn.Embedding(2, hidden_dim)

        # Castling projection
        self.castling_proj = nn.Linear(4, hidden_dim)

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    @property
    def name(self) -> str:
        return f"GCN_{self.num_layers}L_{self.hidden_dim}D"

    def get_backbone_output_dim(self) -> int:
        return self.output_dim

    def forward_backbone(self, x: torch.Tensor | dict) -> torch.Tensor:
        assert isinstance(x, dict), "GCN expects a dict input"
        node_features = x["x"]
        edge_index = x["edge_index"]
        side_to_move = x["side_to_move"]
        castling = x["castling"]

        if "batch" in x:
            batch = x["batch"].to(node_features.device)
        elif node_features.dim() == 3:
            batch_size = node_features.size(0)
            node_features = node_features.view(-1, node_features.size(-1))
            batch = torch.arange(batch_size, device=node_features.device)
            batch = batch.unsqueeze(1).expand(-1, 64).reshape(-1)
            edge_indices = []
            for i in range(batch_size):
                edge_indices.append(edge_index + i * 64)
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            batch = torch.zeros(
                node_features.size(0), dtype=torch.long, device=node_features.device
            )

        h = self.input_proj(node_features)
        h = self.dropout(F.relu(h))

        for layer in self.gcn_layers:
            h = layer(h, edge_index)

        output = global_mean_pool(h, batch)

        side_emb = self.side_embedding(side_to_move)
        output = output + side_emb

        castling_emb = self.castling_proj(castling)
        output = output + castling_emb

        output = self.output_proj(output)

        return cast(torch.Tensor, output)
