"""
GNN State Encoder - Outputs graph data objects with rich edge features.
"""

import chess
import numpy as np
import torch

from .base import StateEncoder


class GNNEncoder(StateEncoder):
    """
    GNN state encoder for graph neural network processing.

    Nodes: 64 squares
    Node features:
        - Piece ID one-hot: 13 values (Empty, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK)
        - Color: 1 value (0=Empty, 1=White, -1=Black) - optional, redundant with ID but requested

    Edges:
        - Adjacency matrix (Undirected)
        - Edge Features (for GAT):
            - Distance: Manhattan distance (normalized)
            - Ray-tracing: Is path clear? (1/0)
            - Legality: Is move legal? (1/0)
    """  # noqa: E501

    NODE_FEATURE_DIM = 14  # 13 (Piece ID) + 1 (Color)
    EDGE_FEATURE_DIM = 3  # Distance, Ray-tracing, Legality

    def __init__(self, edge_type: str = "hybrid"):
        self.edge_type = edge_type
        self._name = f"GNNEncoder_{edge_type}"

        self._static_neighbors: list[list[int]] = [[] for _ in range(64)]
        self._static_edges: list[tuple[int, int]] = []
        self._static_dists: list[float] = []
        for sq in range(64):
            row, col = divmod(sq, 8)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        target = nr * 8 + nc
                        self._static_neighbors[sq].append(target)
                        dist = (abs(dr) + abs(dc)) / 14.0
                        self._static_edges.append((sq, target))
                        self._static_dists.append(dist)

    @property
    def name(self) -> str:
        return self._name

    def get_input_shape(self) -> tuple:
        return (64, self.NODE_FEATURE_DIM)

    def encode(self, board: chess.Board) -> dict:
        x = self._build_node_features(board)

        legal_set: set[tuple[int, int]] = {
            (m.from_square, m.to_square) for m in board.legal_moves
        }

        seen_edges: set[tuple[int, int]] = set()
        edges: list[list[int]] = []
        edge_attrs: list[list[float]] = []

        def add_edge(u: int, v: int, dist: float):
            if (u, v) in seen_edges:
                return
            seen_edges.add((u, v))
            legality = 1.0 if (u, v) in legal_set else 0.0
            edges.append([u, v])
            edge_attrs.append([dist, 1.0, legality])

        for (sq, target), dist in zip(
            self._static_edges, self._static_dists, strict=False
        ):
            add_edge(sq, target, dist)

        for sq in range(64):
            attackers = board.attackers(chess.WHITE, sq) | board.attackers(
                chess.BLACK, sq
            )
            for attacker_sq in attackers:
                u_row, u_col = divmod(attacker_sq, 8)
                v_row, v_col = divmod(sq, 8)
                dist = (abs(u_row - v_row) + abs(u_col - v_col)) / 14.0
                add_edge(attacker_sq, sq, dist)

        final_edges: list[list[int]] = []
        final_attrs: list[list[float]] = []
        edge_map: dict[tuple[int, int], list[float]] = {}
        for i, (u, v) in enumerate(edges):
            edge_map[(u, v)] = edge_attrs[i]

        reverse_additions: list[tuple[tuple[int, int], list[float]]] = []
        for (u, v), attr in edge_map.items():
            final_edges.append([u, v])
            final_attrs.append(attr)

            if (v, u) not in edge_map:
                rev_legality = 1.0 if (v, u) in legal_set else 0.0
                rev_attr = [attr[0], attr[1], rev_legality]
                reverse_additions.append(((v, u), rev_attr))

        for (u, v), attr in reverse_additions:
            if (u, v) not in edge_map:
                final_edges.append([u, v])
                final_attrs.append(attr)
                edge_map[(u, v)] = attr

        return {
            "x": x,
            "edge_index": torch.tensor(final_edges, dtype=torch.long).t().contiguous(),
            "edge_attr": torch.tensor(final_attrs, dtype=torch.float32),
            "side_to_move": torch.tensor(
                0 if board.turn == chess.WHITE else 1, dtype=torch.long
            ),
            "castling": torch.from_numpy(self._get_castling(board)),
        }

    def _build_node_features(self, board: chess.Board) -> torch.Tensor:
        # 13 Piece IDs + 1 Color
        features = np.zeros((64, self.NODE_FEATURE_DIM), dtype=np.float32)

        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is None:
                features[sq, 0] = 1.0  # Empty ID
                features[sq, 13] = 0.0  # Color 0
            else:
                # Map piece to 1-12
                # White: 1-6 (P,N,B,R,Q,K)
                # Black: 7-12
                offset = 0 if piece.color == chess.WHITE else 6
                idx = piece.piece_type + offset  # 1..6 or 7..12
                features[sq, idx] = 1.0

                # Color
                features[sq, 13] = 1.0 if piece.color == chess.WHITE else -1.0

        return torch.from_numpy(features)

    def _get_castling(self, board):
        return np.array(
            [
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK),
            ],
            dtype=np.float32,
        )
