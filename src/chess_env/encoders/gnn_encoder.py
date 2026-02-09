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
    """
    
    NODE_FEATURE_DIM = 14  # 13 (Piece ID) + 1 (Color)
    EDGE_FEATURE_DIM = 3   # Distance, Ray-tracing, Legality
    
    def __init__(self, edge_type: str = "hybrid"):
        self.edge_type = edge_type
        self._name = f"GNNEncoder_{edge_type}"
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_input_shape(self) -> tuple:
        return (64, self.NODE_FEATURE_DIM)
    
    def encode(self, board: chess.Board) -> dict:
        """
        Encode the board as a graph.
        
        Returns:
            Dictionary with:
                - 'x': Node features (64, 14)
                - 'edge_index': Adjacency list (2, E)
                - 'edge_attr': Edge features (E, 3)
                - 'side_to_move': ...
                - 'castling': ...
        """
        # Node Features
        x = self._build_node_features(board)
        
        # Edges
        # We build a dense adjacency concept first then convert to sparse
        # But for efficiency, let's iterate.
        # User defined neighbors:
        # 1. Physically adjacent (King moves)
        # 2. Attack/Defend relations
        
        # Collect edges
        edges = []
        edge_attrs = []
        
        # 1. Physical Adjacency (King moves)
        # Undirected, so we add (u, v) and (v, u).
        # To avoid valid duplicates with attack edges, we track seen edges.
        seen_edges = set()
        
        def add_edge(u, v, is_legal):
            if u == v: return
            if (u, v) in seen_edges: return
            
            seen_edges.add((u, v))
            edges.append([u, v])
            
            # Features
            u_row, u_col = divmod(u, 8)
            v_row, v_col = divmod(v, 8)
            
            # Distance (Manhattan normalized)
            dist = (abs(u_row - v_row) + abs(u_col - v_col)) / 14.0
            
            # Ray-tracing (Is blocked?)
            # chess.Board has generic attack check, but for ray tracing we can check if it's a sliding piece relationship
            # For simplicity:
            # If adjacent, not blocked (1.0).
            # If not adjacent, check occupancy between.
            # Using chess.ray_index logic implies knowing direction.
            # Simplified: 1 if "influencing directly", 0 if blocked. 
            # If it is a legal move or attack, it is by definition not blocked.
            # If it's just physical adjacency, it's not blocked.
            # So ray_tracing = 1 if (Legal OR Attack OR Adjacent) else 0 (but we only add these edges anyway)
            # Actually, "blocked by another piece" is interesting for Queens looking at Kings across board.
            # If we only add edges where A attacks/defends B, then it is NOT blocked.
            ray_trace = 1.0 
            
            # Legality
            legality = 1.0 if is_legal else 0.0
            
            edge_attrs.append([dist, ray_trace, legality])

        # Add physical neighbors (King moves)
        for sq in range(64):
            row, col = divmod(sq, 8)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        target = nr * 8 + nc
                        # Is this move legal?
                        move = chess.Move(sq, target)
                        is_legal = board.is_legal(move)
                        add_edge(sq, target, is_legal)

        # 2. Attacks/Defenses
        # Iterating all pieces and their attacks
        # "A attacks B" -> Undirected means A-B edge exists.
        # "A defends B" -> Same as attack but own color.
        
        # We can use board.attackers()
        for sq in range(64):
            # For every square, finding attackers (both colors)
            # This covers "Attacks" and "Defends"
            # If X attacks Y, then X and Y are connected.
            
            attackers = board.attackers(chess.WHITE, sq) | board.attackers(chess.BLACK, sq)
            for attacker_sq in attackers:
                # Check legality (only if attacker is side to move and it's a valid move)
                # But "attack" doesn't imply legality (e.g. pinned piece).
                move = chess.Move(attacker_sq, sq)
                is_legal = board.is_legal(move)
                add_edge(attacker_sq, sq, is_legal)
                # Undirected: add reverse too?
                # GCNs usually aggregate across the whole graph equally, the adjacency matrix should be undirected
                # So if A attacks B, B is a neighbor of A.
                # My add_edge handles unique (u, v). If I add (u, v), I should probably add (v, u) for undirected GNN.
                # The prompt says "Undirected". So I must ensure symmetric adjacency.
                # However, edge *features* might differ (A attacks B is a legal move for A, but B-A might not be).
                # But GCN aggregates "equally".
                # Let's add symmetric edges in a post-processing or modify add_edge to add both if not present.
        
        # Enforce symmetry for undirected graph
        # Currently 'edges' list tracks directed edges (u->v).
        # We need to ensure if u->v exists, v->u exists.
        # But features might differ? "Move Legality" is directional.
        # If A attacks B, A->B is legal (maybe), B->A is likely illegal.
        # User says "Undirected... if A attacks B, B is a neighbor of A".
        # This implies the *structure* is undirected.
        # GAT accepts directed edges but learns weights.
        # If I want true undirected behavior in GCN, I usually treat edges as symmetric.
        # For GAT with edge features, directional features (legality) are valuable.
        
        # Compromise: Build a directed graph including "reverse" edges where attributes allow.
        # If A->B is legal, B->A (reverse edge) has [Dist, Ray, 0.0] (Illegal).
        
        # Let's verify existing edges and add reverses if missing.
        final_edges = []
        final_attrs = []
        
        # Convert to dict for fast lookup
        edge_map = {}
        for i, (u, v) in enumerate(edges):
            edge_map[(u, v)] = edge_attrs[i]
            
        # Add keys to a list to iterate safely
        keys = list(edge_map.keys())
        for u, v in keys:
            final_edges.append([u, v])
            final_attrs.append(edge_map[(u, v)])
            
            if (v, u) not in edge_map:
                # Add reverse edge
                # Dist is same
                # Ray trace is same (line of sight is symmetric)
                # Legality: Check if v->u is legal
                move = chess.Move(v, u)
                is_legal = board.is_legal(move)
                attr = edge_map[(u, v)]
                new_attr = [attr[0], attr[1], 1.0 if is_legal else 0.0]
                
                final_edges.append([v, u])
                final_attrs.append(new_attr)
                edge_map[(v, u)] = new_attr

        return {
            'x': x,
            'edge_index': torch.tensor(final_edges, dtype=torch.long).t().contiguous(),
            'edge_attr': torch.tensor(final_attrs, dtype=torch.float32),
            'side_to_move': torch.tensor(0 if board.turn == chess.WHITE else 1, dtype=torch.long),
            'castling': torch.from_numpy(self._get_castling(board))
        }

    def _build_node_features(self, board: chess.Board) -> torch.Tensor:
        # 13 Piece IDs + 1 Color
        features = np.zeros((64, self.NODE_FEATURE_DIM), dtype=np.float32)
        
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is None:
                features[sq, 0] = 1.0 # Empty ID
                features[sq, 13] = 0.0 # Color 0
            else:
                # Map piece to 1-12
                # White: 1-6 (P,N,B,R,Q,K)
                # Black: 7-12
                offset = 0 if piece.color == chess.WHITE else 6
                idx = piece.piece_type + offset # 1..6 or 7..12
                features[sq, idx] = 1.0
                
                # Color
                features[sq, 13] = 1.0 if piece.color == chess.WHITE else -1.0
                
        return torch.from_numpy(features)

    def _get_castling(self, board):
        return np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ], dtype=np.float32)

