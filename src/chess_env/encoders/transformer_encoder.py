"""
Transformer State Encoder - Outputs token sequences.

Two tokenization schemes:
1. SquareTokenizer: 64 fixed tokens (one per square)
2. PieceTokenizer: Variable-length (one token per piece, max 32)
"""
import chess
import numpy as np
import torch

from .base import StateEncoder


class SquareTokenizer:
    """
    Tokenizes the board as 64 tokens (one per square).
    
    Each token encodes:
        - Piece type (0-6, where 0 = empty)
        - Piece color (0-1)
        - Square index (0-63) as positional info
    """
    
    # Token vocabulary
    # 0: empty, 1-6: white pieces (P,N,B,R,Q,K), 7-12: black pieces
    VOCAB_SIZE = 13
    NUM_SQUARES = 64
    
    # Additional features per token
    FEATURE_DIM = 8  # piece_type(7) + color(1)
    
    PIECE_TO_TOKEN = {
        None: 0,
        (chess.PAWN, chess.WHITE): 1,
        (chess.KNIGHT, chess.WHITE): 2,
        (chess.BISHOP, chess.WHITE): 3,
        (chess.ROOK, chess.WHITE): 4,
        (chess.QUEEN, chess.WHITE): 5,
        (chess.KING, chess.WHITE): 6,
        (chess.PAWN, chess.BLACK): 7,
        (chess.KNIGHT, chess.BLACK): 8,
        (chess.BISHOP, chess.BLACK): 9,
        (chess.ROOK, chess.BLACK): 10,
        (chess.QUEEN, chess.BLACK): 11,
        (chess.KING, chess.BLACK): 12,
    }
    
    def tokenize(self, board: chess.Board) -> dict:
        """
        Tokenize the board into 64 tokens.
        
        Args:
            board: A python-chess Board object.
        
        Returns:
            Dictionary with:
                - 'tokens': Token IDs of shape (64,)
                - 'positions': Position indices of shape (64,)
                - 'attention_mask': All ones of shape (64,)
                - 'side_to_move': 0 for white, 1 for black
                - 'castling': 4-element tensor for castling rights
        """
        tokens = np.zeros(64, dtype=np.int64)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                tokens[square] = 0
            else:
                tokens[square] = self.PIECE_TO_TOKEN[(piece.piece_type, piece.color)]
        
        # Position indices (always 0-63)
        positions = np.arange(64, dtype=np.int64)
        
        # Attention mask (all ones for square tokenizer)
        attention_mask = np.ones(64, dtype=np.float32)
        
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
            'tokens': torch.from_numpy(tokens),
            'positions': torch.from_numpy(positions),
            'attention_mask': torch.from_numpy(attention_mask),
            'side_to_move': torch.tensor(side_to_move, dtype=torch.long),
            'castling': torch.from_numpy(castling),
        }


class PieceTokenizer:
    """
    Tokenizes the board as variable-length sequence (one token per piece).
    
    Each token encodes:
        - Piece type and color
        - Square position
    
    Maximum sequence length: 32 (all pieces on board)
    """
    
    VOCAB_SIZE = 12  # 6 piece types × 2 colors
    MAX_PIECES = 32
    
    PIECE_TO_TOKEN = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    def tokenize(self, board: chess.Board) -> dict:
        """
        Tokenize the board into variable-length piece sequence.
        
        Args:
            board: A python-chess Board object.
        
        Returns:
            Dictionary with:
                - 'tokens': Token IDs of shape (max_pieces,)
                - 'positions': Square positions of shape (max_pieces,)
                - 'attention_mask': Valid positions mask of shape (max_pieces,)
                - 'side_to_move': 0 for white, 1 for black
                - 'castling': 4-element tensor for castling rights
                - 'num_pieces': Actual number of pieces
        """
        tokens = np.zeros(self.MAX_PIECES, dtype=np.int64)
        positions = np.zeros(self.MAX_PIECES, dtype=np.int64)
        attention_mask = np.zeros(self.MAX_PIECES, dtype=np.float32)
        
        piece_idx = 0
        
        # Collect all pieces (our pieces first, then opponent's)
        if board.turn == chess.WHITE:
            colors = [chess.WHITE, chess.BLACK]
        else:
            colors = [chess.BLACK, chess.WHITE]
        
        for color in colors:
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None and piece.color == color:
                    if piece_idx < self.MAX_PIECES:
                        tokens[piece_idx] = self.PIECE_TO_TOKEN[(piece.piece_type, piece.color)]
                        positions[piece_idx] = square
                        attention_mask[piece_idx] = 1.0
                        piece_idx += 1
        
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
            'tokens': torch.from_numpy(tokens),
            'positions': torch.from_numpy(positions),
            'attention_mask': torch.from_numpy(attention_mask),
            'side_to_move': torch.tensor(side_to_move, dtype=torch.long),
            'castling': torch.from_numpy(castling),
            'num_pieces': torch.tensor(piece_idx, dtype=torch.long),
        }


class TransformerEncoder(StateEncoder):
    """
    Transformer state encoder supporting both tokenization schemes.
    """
    
    def __init__(self, tokenizer_type: str = "square"):
        """
        Initialize the transformer encoder.
        
        Args:
            tokenizer_type: Either "square" (64 tokens) or "piece" (variable-length).
        """
        if tokenizer_type == "square":
            self.tokenizer = SquareTokenizer()
            self._name = "SquareTransformerEncoder"
        elif tokenizer_type == "piece":
            self.tokenizer = PieceTokenizer()
            self._name = "PieceTransformerEncoder"
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        self.tokenizer_type = tokenizer_type
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_input_shape(self) -> tuple:
        """
        Get the shape information for the transformer input.
        
        Returns:
            For square: (64, vocab_size=13)
            For piece: (32, vocab_size=12)
        """
        if self.tokenizer_type == "square":
            return (64, SquareTokenizer.VOCAB_SIZE)
        else:
            return (32, PieceTokenizer.VOCAB_SIZE)
    
    def encode(self, board: chess.Board) -> dict:
        """
        Encode the board as a token sequence.
        
        Args:
            board: A python-chess Board object.
        
        Returns:
            Dictionary containing token tensors.
        """
        return self.tokenizer.tokenize(board)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.VOCAB_SIZE
    
    @property
    def max_seq_length(self) -> int:
        """Get the maximum sequence length."""
        if self.tokenizer_type == "square":
            return 64
        else:
            return PieceTokenizer.MAX_PIECES
