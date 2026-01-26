"""
GNN State Encoder - Outputs graph data objects.

Three edge types:
1. StaticEdges: Spatial adjacency (king-move pattern)
2. DynamicEdges: Current legal moves as edges
3. HybridEdges: Both edge types (heterogeneous graph)
"""
import chess
import numpy as np
import torch

from .base import StateEncoder


class StaticEdgeBuilder:
    """
    Builds static edges based on spatial adjacency (king-move pattern).
    
    Each square is connected to its 8 neighbors (or fewer at edges).
    This creates a fixed graph structure that doesn't change during the game.
    """
    
    def __init__(self):
        # Pre-compute static edges (same for all positions)
        self.edge_index = self._compute_static_edges()
    
    def _compute_static_edges(self) -> torch.Tensor:
        """Compute king-move adjacency edges."""
        edges = []
        
        for square in range(64):
            row = square // 8
            col = square % 8
            
            # All 8 directions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    new_row = row + dr
                    new_col = col + dc
                    
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target = new_row * 8 + new_col
                        edges.append([square, target])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def build(self, board: chess.Board) -> torch.Tensor:
        """
        Build static edges (same for all positions).
        
        Args:
            board: Chess board (unused, edges are static).
        
        Returns:
            Edge index tensor of shape (2, num_edges).
        """
        return self.edge_index


class DynamicEdgeBuilder:
    """
    Builds dynamic edges based on current legal moves.
    
    Each legal move creates an edge from the source square to the target square.
    This captures the current tactical possibilities in the position.
    """
    
    def build(self, board: chess.Board) -> torch.Tensor:
        """
        Build edges from legal moves.
        
        Args:
            board: Chess board.
        
        Returns:
            Edge index tensor of shape (2, num_edges).
        """
        edges = []
        
        for move in board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            edges.append([from_sq, to_sq])
        
        if len(edges) == 0:
            # No legal moves (checkmate or stalemate)
            return torch.zeros((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class HybridEdgeBuilder:
    """
    Builds both static and dynamic edges as a heterogeneous graph.
    
    Returns separate edge indices for:
    - 'spatial': King-move adjacency edges
    - 'legal': Current legal move edges
    """
    
    def __init__(self):
        self.static_builder = StaticEdgeBuilder()
        self.dynamic_builder = DynamicEdgeBuilder()
    
    def build(self, board: chess.Board) -> dict[str, torch.Tensor]:
        """
        Build both edge types.
        
        Args:
            board: Chess board.
        
        Returns:
            Dictionary with 'spatial' and 'legal' edge indices.
        """
        return {
            'spatial': self.static_builder.build(board),
            'legal': self.dynamic_builder.build(board),
        }


class GNNEncoder(StateEncoder):
    """
    GNN state encoder for graph neural network processing.
    
    Nodes: 64 squares
    Node features:
        - Piece type one-hot (7 values: empty + 6 pieces)
        - Piece color (1 value)
        - Square position encoding (row and column, 2 values)
        - Is attacked by us (1 value)
        - Is attacked by them (1 value)
        - Total: 12 features per node
    
    Edges: Configurable (static, dynamic, or hybrid)
    """
    
    NODE_FEATURE_DIM = 12
    
    def __init__(self, edge_type: str = "hybrid"):
        """
        Initialize the GNN encoder.
        
        Args:
            edge_type: "static", "dynamic", or "hybrid"
        """
        self.edge_type = edge_type
        
        if edge_type == "static":
            self.edge_builder = StaticEdgeBuilder()
        elif edge_type == "dynamic":
            self.edge_builder = DynamicEdgeBuilder()
        elif edge_type == "hybrid":
            self.edge_builder = HybridEdgeBuilder()
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")
    
    @property
    def name(self) -> str:
        return f"GNNEncoder_{self.edge_type}"
    
    def get_input_shape(self) -> tuple:
        return (64, self.NODE_FEATURE_DIM)
    
    def encode(self, board: chess.Board) -> dict:
        """
        Encode the board as a graph.
        
        Args:
            board: A python-chess Board object.
        
        Returns:
            Dictionary with:
                - 'x': Node features of shape (64, 12)
                - 'edge_index': Edge indices (format depends on edge_type)
                - 'side_to_move': 0 for white, 1 for black
                - 'castling': 4-element tensor for castling rights
        """
        # Build node features
        x = self._build_node_features(board)
        
        # Build edges
        if self.edge_type == "hybrid":
            edge_index = self.edge_builder.build(board)
        else:
            edge_index = self.edge_builder.build(board)
        
        # Side to move
        side_to_move = 0 if board.turn == chess.WHITE else 1
        
        # Castling rights
        castling = np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ], dtype=np.float32)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'side_to_move': torch.tensor(side_to_move, dtype=torch.long),
            'castling': torch.from_numpy(castling),
        }
    
    def _build_node_features(self, board: chess.Board) -> torch.Tensor:
        """
        Build node feature matrix.
        
        Features per node (64 nodes total):
            - Piece type one-hot: 7 values (empty, pawn, knight, bishop, rook, queen, king)
            - Piece color: 1 value (0=empty/white, 1=black, 0.5=empty)
            - Position encoding: 2 values (row/8, col/8 normalized)
            - Attacked by us: 1 value
            - Attacked by them: 1 value
        
        Total: 12 features
        """
        features = np.zeros((64, self.NODE_FEATURE_DIM), dtype=np.float32)
        
        us = board.turn
        them = not us
        
        for square in range(64):
            piece = board.piece_at(square)
            
            # Piece type one-hot (indices 0-6)
            if piece is None:
                features[square, 0] = 1.0  # Empty
            else:
                features[square, piece.piece_type] = 1.0  # piece_type is 1-6
            
            # Piece color (index 7)
            if piece is None:
                features[square, 7] = 0.5  # Neutral for empty
            elif piece.color == chess.WHITE:
                features[square, 7] = 0.0
            else:
                features[square, 7] = 1.0
            
            # Position encoding (indices 8-9)
            row = square // 8
            col = square % 8
            features[square, 8] = row / 7.0  # Normalize to [0, 1]
            features[square, 9] = col / 7.0
            
            # Attacked by us (index 10)
            features[square, 10] = 1.0 if board.is_attacked_by(us, square) else 0.0
            
            # Attacked by them (index 11)
            features[square, 11] = 1.0 if board.is_attacked_by(them, square) else 0.0
        
        return torch.from_numpy(features)
