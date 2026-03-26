"""
Square-based Transformer (ChessFormer) - 64 fixed tokens.
"""

from typing import cast

import torch
import torch.nn as nn

from ..base import ChessModel


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for chess board positions."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings."""
        return cast(torch.Tensor, x + self.pos_embedding(positions))


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=attention_mask)
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))

        return x


class SquareTransformer(ChessModel):
    """
    Square-based ChessFormer with 64 fixed tokens (one per square).

    Architecture:
        - Token embedding (13 vocab: empty + 12 pieces)
        - Learnable positional embeddings (64 positions)
        - N Transformer encoder layers
        - [CLS] token aggregation or mean pooling
        - Output dimension: embed_dim
    """

    def __init__(
        self,
        vocab_size: int = 13,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        """
        Initialize SquareTransformer.

        Args:
            vocab_size: Token vocabulary size (13 for square tokenizer).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of Transformer layers.
            dropout: Dropout rate.
            use_cls_token: If True, use CLS token; else mean pooling.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_cls_token = use_cls_token

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding (64 squares + optional CLS token)
        max_positions = 65 if use_cls_token else 64
        self.pos_embedding = nn.Embedding(max_positions, embed_dim)

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # Side to move embedding
        self.side_embedding = nn.Embedding(2, embed_dim)

        # Castling embedding
        self.castling_proj = nn.Linear(4, embed_dim)

        # Coordinate projection
        self.coord_proj = nn.Linear(2, embed_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.output_dim = embed_dim

    @property
    def name(self) -> str:
        return f"SquareTransformer_{self.num_layers}L_{self.embed_dim}D"

    def get_backbone_output_dim(self) -> int:
        return self.output_dim

    def forward_backbone(self, x: torch.Tensor | dict) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Dictionary with:
                - 'tokens': Token IDs of shape (B, 64)
                - 'positions': Position indices of shape (B, 64)
                - 'attention_mask': Attention mask of shape (B, 64)
                - 'side_to_move': Side to move of shape (B,)
                - 'castling': Castling rights of shape (B, 4)

        Returns:
            Feature tensor of shape (B, embed_dim).
        """
        assert isinstance(x, dict), "SquareTransformer expects a dict input"
        tokens = x["tokens"]  # (B, 64)
        positions = x["positions"]  # (B, 64)
        coordinates = x.get("coordinates")  # (B, 64, 2) - Normalized rank/file
        side_to_move = x["side_to_move"]  # (B,)
        castling = x["castling"]  # (B, 4)

        batch_size = tokens.size(0)

        # Token embeddings
        token_emb = self.token_embedding(tokens)  # (B, 64, D)

        # Add positional embeddings
        pos_emb = self.pos_embedding(positions + 1) if self.use_cls_token else self.pos_embedding(positions)

        embeddings = token_emb + pos_emb  # (B, 64, D)

        # Add coordinate embeddings if available
        if coordinates is not None:
            coord_emb = self.coord_proj(coordinates)
            embeddings = embeddings + coord_emb

        # Add side to move as bias to all tokens
        side_emb = self.side_embedding(side_to_move)  # (B, D)
        embeddings = embeddings + side_emb.unsqueeze(1)

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
            # Add castling info to CLS token
            castling_emb = self.castling_proj(castling).unsqueeze(1)  # (B, 1, D)
            cls_tokens = cls_tokens + castling_emb
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (B, 65, D)

        embeddings = self.dropout(embeddings)

        # Transformer layers
        hidden = embeddings
        for layer in self.transformer_layers:
            hidden = layer(hidden)

        hidden = self.norm(hidden)

        # Output aggregation
        # CLS token output (B, D)
        # Mean pooling (B, D)
        output = hidden[:, 0] if self.use_cls_token else hidden.mean(dim=1)

        return cast(torch.Tensor, output)
