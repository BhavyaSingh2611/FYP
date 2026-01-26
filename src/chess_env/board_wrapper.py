"""
Chess board wrapper using python-chess.
"""
import chess
import numpy as np
from typing import Optional


# All possible UCI moves in chess (for policy output)
# This includes all possible promotions and all possible moves
def _generate_all_uci_moves() -> list[str]:
    """Generate all possible UCI move strings."""
    moves = []
    files = 'abcdefgh'
    ranks = '12345678'
    promotion_pieces = ['q', 'r', 'b', 'n']
    
    for from_file in files:
        for from_rank in ranks:
            for to_file in files:
                for to_rank in ranks:
                    from_sq = from_file + from_rank
                    to_sq = to_file + to_rank
                    
                    if from_sq == to_sq:
                        continue
                    
                    # Regular move
                    moves.append(from_sq + to_sq)
                    
                    # Pawn promotions (from rank 7 to 8 for white, 2 to 1 for black)
                    if (from_rank == '7' and to_rank == '8') or \
                       (from_rank == '2' and to_rank == '1'):
                        for promo in promotion_pieces:
                            moves.append(from_sq + to_sq + promo)
    
    return moves


ALL_UCI_MOVES = _generate_all_uci_moves()
UCI_MOVE_TO_INDEX = {move: idx for idx, move in enumerate(ALL_UCI_MOVES)}
INDEX_TO_UCI_MOVE = {idx: move for idx, move in enumerate(ALL_UCI_MOVES)}
NUM_MOVES = len(ALL_UCI_MOVES)


class BoardWrapper:
    """
    Wrapper around python-chess Board with additional functionality.
    """
    
    def __init__(self, fen: Optional[str] = None):
        """
        Initialize the board wrapper.
        
        Args:
            fen: Optional FEN string to initialize the board.
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
    
    @property
    def fen(self) -> str:
        """Get the current FEN string."""
        return self.board.fen()
    
    @property
    def turn(self) -> bool:
        """Get the side to move (True = White, False = Black)."""
        return self.board.turn
    
    @property
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
    
    @property
    def result(self) -> str:
        """Get the game result if over."""
        return self.board.result()
    
    @property
    def legal_moves(self) -> list[chess.Move]:
        """Get all legal moves."""
        return list(self.board.legal_moves)
    
    @property
    def legal_moves_uci(self) -> list[str]:
        """Get all legal moves as UCI strings."""
        return [move.uci() for move in self.board.legal_moves]
    
    def push(self, move: chess.Move | str) -> None:
        """
        Make a move on the board.
        
        Args:
            move: Move object or UCI string.
        """
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        self.board.push(move)
    
    def pop(self) -> chess.Move:
        """Undo the last move."""
        return self.board.pop()
    
    def copy(self) -> 'BoardWrapper':
        """Create a copy of the board."""
        wrapper = BoardWrapper()
        wrapper.board = self.board.copy()
        return wrapper
    
    def reset(self) -> None:
        """Reset to starting position."""
        self.board.reset()
    
    def set_fen(self, fen: str) -> None:
        """Set the board to a FEN position."""
        self.board.set_fen(fen)
    
    @staticmethod
    def move_to_index(move: chess.Move | str) -> int:
        """
        Convert a move to its index in the policy output.
        
        Args:
            move: Move object or UCI string.
        
        Returns:
            Index in the move space.
        """
        if isinstance(move, chess.Move):
            move = move.uci()
        return UCI_MOVE_TO_INDEX.get(move, -1)
    
    @staticmethod
    def index_to_move(index: int) -> str:
        """
        Convert an index to a UCI move string.
        
        Args:
            index: Index in the move space.
        
        Returns:
            UCI move string.
        """
        return INDEX_TO_UCI_MOVE.get(index, "")
    
    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """Get the piece at a square (0-63)."""
        return self.board.piece_at(square)
    
    def get_all_pieces(self) -> list[tuple[int, chess.Piece]]:
        """
        Get all pieces on the board.
        
        Returns:
            List of (square, piece) tuples.
        """
        pieces = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                pieces.append((square, piece))
        return pieces
    
    def get_attacks_from(self, square: int) -> list[int]:
        """Get all squares attacked from a given square."""
        return list(self.board.attacks(square))
    
    def is_check(self) -> bool:
        """Check if the current side is in check."""
        return self.board.is_check()
    
    def is_checkmate(self) -> bool:
        """Check if the current side is checkmated."""
        return self.board.is_checkmate()
    
    def is_stalemate(self) -> bool:
        """Check if the position is stalemate."""
        return self.board.is_stalemate()
    
    def halfmove_clock(self) -> int:
        """Get the halfmove clock (for 50-move rule)."""
        return self.board.halfmove_clock
    
    def fullmove_number(self) -> int:
        """Get the fullmove number."""
        return self.board.fullmove_number
    
    def has_kingside_castling_rights(self, color: bool) -> bool:
        """Check if a color has kingside castling rights."""
        return self.board.has_kingside_castling_rights(color)
    
    def has_queenside_castling_rights(self, color: bool) -> bool:
        """Check if a color has queenside castling rights."""
        return self.board.has_queenside_castling_rights(color)
    
    def __repr__(self) -> str:
        return f"BoardWrapper(fen='{self.fen}')"
    
    def __str__(self) -> str:
        return str(self.board)
