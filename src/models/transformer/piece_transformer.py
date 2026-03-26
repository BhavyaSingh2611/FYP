"""
Piece-based Transformer - Variable-length sequence (up to 32 pieces).
"""

from typing import cast

import torch
import torch.nn as nn

from ..base import ChessModel


class PieceTransformerBlock(nn.Module):
    """Transformer block with masked attention for variable-length sequences."""

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
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))

        return x


class PieceTransformer(ChessModel):
    """
    Piece-based ChessFormer with variable-length sequence (≤32 pieces).

    Architecture:
        - Piece type embedding (12 vocab: 6 pieces × 2 colors)
        - Position embedding (64 squares)
        - N Transformer encoder layers with padding mask
        - CLS token aggregation
        - Output dimension: embed_dim

    This variant processes only the pieces on the board, leading to
    shorter sequences and potentially more efficient attention.
    """

    def __init__(
        self,
        vocab_size: int = 12,
        max_pieces: int = 32,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        """
        Initialize PieceTransformer.

        Args:
            vocab_size: Piece vocabulary size (12 pieces).
            max_pieces: Maximum number of pieces (32).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of Transformer layers.
            dropout: Dropout rate.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_pieces = max_pieces

        # Piece type embedding
        self.piece_embedding = nn.Embedding(vocab_size, embed_dim)

        # Position embedding (64 squares)
        self.pos_embedding = nn.Embedding(64, embed_dim)

        # CLS token (position 0, piece tokens at positions 1-33)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Sequence position embedding (for ordering)
        self.seq_pos_embedding = nn.Embedding(max_pieces + 1, embed_dim)  # +1 for CLS

        # Side to move embedding
        self.side_embedding = nn.Embedding(2, embed_dim)

        # Castling embedding
        self.castling_proj = nn.Linear(4, embed_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                PieceTransformerBlock(embed_dim, num_heads, dropout=dropout)
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
        return f"PieceTransformer_{self.num_layers}L_{self.embed_dim}D"

    def get_backbone_output_dim(self) -> int:
        return self.output_dim

    def forward_backbone(self, x: torch.Tensor | dict) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Dictionary with:
                - 'tokens': Piece type IDs of shape (B, max_pieces)
                - 'positions': Square positions of shape (B, max_pieces)
                - 'attention_mask': Valid positions mask of shape (B, max_pieces)
                - 'side_to_move': Side to move of shape (B,)
                - 'castling': Castling rights of shape (B, 4)

        Returns:
            Feature tensor of shape (B, embed_dim).
        """
        assert isinstance(x, dict), "PieceTransformer expects a dict input"
        tokens = x["tokens"]  # (B, 32)
        positions = x["positions"]  # (B, 32) - square positions
        attention_mask = x["attention_mask"]  # (B, 32) - 1 for valid, 0 for padding
        side_to_move = x["side_to_move"]  # (B,)
        castling = x["castling"]  # (B, 4)

        batch_size = tokens.size(0)
        device = tokens.device

        # Piece embeddings
        piece_emb = self.piece_embedding(tokens)  # (B, 32, D)

        # Add square position embeddings
        pos_emb = self.pos_embedding(positions)  # (B, 32, D)

        # Add sequence position embeddings (token order: 1-32)
        seq_positions = torch.arange(1, self.max_pieces + 1, device=device)
        seq_positions = seq_positions.unsqueeze(0).expand(batch_size, -1)  # (B, 32)
        seq_pos_emb = self.seq_pos_embedding(seq_positions)  # (B, 32, D)

        embeddings = piece_emb + pos_emb + seq_pos_emb  # (B, 32, D)

        # Add side to move as bias to all tokens
        side_emb = self.side_embedding(side_to_move)  # (B, D)
        embeddings = embeddings + side_emb.unsqueeze(1)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)

        # Add CLS position embedding (position 0)
        cls_pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_pos_emb = self.seq_pos_embedding(cls_pos)  # (B, 1, D)
        cls_tokens = cls_tokens + cls_pos_emb

        # Add castling info to CLS token
        castling_emb = self.castling_proj(castling).unsqueeze(1)  # (B, 1, D)
        cls_tokens = cls_tokens + castling_emb

        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (B, 33, D)
        embeddings = self.dropout(embeddings)

        # Create padding mask for attention (True = masked/padding)
        # CLS token is never masked
        cls_mask = torch.zeros(batch_size, 1, device=device)
        key_padding_mask = torch.cat([cls_mask, 1 - attention_mask], dim=1)  # (B, 33)
        key_padding_mask = key_padding_mask.bool()

        # Transformer layers
        hidden = embeddings
        for layer in self.transformer_layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)

        hidden = self.norm(hidden)

        # CLS token output
        output = hidden[:, 0]  # (B, D)

        return cast(torch.Tensor, output)
