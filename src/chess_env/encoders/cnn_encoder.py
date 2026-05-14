"""
CNN State Encoder - Outputs 8x8xN tensor (bitboard representation).
"""

import chess
import numpy as np
import torch

from .base import StateEncoder


class CNNEncoder(StateEncoder):
    """
    Encodes chess board as an 8x8xN tensor for CNN processing.

    Channels:
        - 12 channels: Piece positions (6 piece types × 2 colors)
        - 4 channels: Castling rights (WK, WQ, BK, BQ)
        - 1 channel: En passant square
        - 1 channel: Side to move

    Total: 18 channels

    The board is always encoded from the perspective of the current player,
    with their pieces in the first 6 channels.
    """

    NUM_CHANNELS = 18

    # Piece type indices (0-5)
    PIECE_TYPES = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    def __init__(self, flip_perspective: bool = True):
        """
        Initialize the CNN encoder.

        Args:
            flip_perspective: If True, always encode from current player's perspective.
        """
        self.flip_perspective = flip_perspective

    @property
    def name(self) -> str:
        return "CNNEncoder"

    def get_input_shape(self) -> tuple:
        return (self.NUM_CHANNELS, 8, 8)

    def encode(self, board: chess.Board) -> torch.Tensor:
        """
        Encode the board as an 8x8x18 tensor.

        Args:
            board: A python-chess Board object.

        Returns:
            torch.Tensor of shape (18, 8, 8).
        """
        tensor = np.zeros((self.NUM_CHANNELS, 8, 8), dtype=np.float32)

        # Determine perspective
        if self.flip_perspective and not board.turn:
            # Black to move - flip the board
            us = chess.BLACK
            them = chess.WHITE
            flip = True
        else:
            us = chess.WHITE
            them = chess.BLACK
            flip = False

        piece_type_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        for square, piece in board.piece_map().items():
            row, col = self._square_to_coords(square, flip)
            channel = piece_type_to_channel[piece.piece_type]
            if piece.color != us:
                channel += 6
            tensor[channel, row, col] = 1.0

        # Encode castling rights (channels 12-15)
        if board.has_kingside_castling_rights(us):
            tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(us):
            tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(them):
            tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(them):
            tensor[15, :, :] = 1.0

        # Encode en passant square (channel 16)
        if board.ep_square is not None:
            row, col = self._square_to_coords(board.ep_square, flip)
            tensor[16, row, col] = 1.0

        # Encode side to move (channel 17)
        # 1.0 for White, -1.0 for Black
        side_val = 1.0 if board.turn == chess.WHITE else -1.0
        tensor[17, :, :] = side_val

        return torch.from_numpy(tensor)

    def encode_batch(self, boards: list[chess.Board]) -> torch.Tensor:
        n = len(boards)
        tensor = np.zeros((n, self.NUM_CHANNELS, 8, 8), dtype=np.float32)

        piece_type_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for b_idx, board in enumerate(boards):
            if self.flip_perspective and not board.turn:
                us, flip = chess.BLACK, True
            else:
                us, flip = chess.WHITE, False

            for square, piece in board.piece_map().items():
                row = 7 - (square // 8)
                col = square % 8
                if flip:
                    row = 7 - row
                    col = 7 - col
                channel = piece_type_to_channel[piece.piece_type]

                if piece.color != us:
                    channel += 6

                tensor[b_idx, channel, row, col] = 1.0

            them = chess.BLACK if us == chess.WHITE else chess.WHITE

            if board.has_kingside_castling_rights(us):
                tensor[b_idx, 12, :, :] = 1.0
            if board.has_queenside_castling_rights(us):
                tensor[b_idx, 13, :, :] = 1.0
            if board.has_kingside_castling_rights(them):
                tensor[b_idx, 14, :, :] = 1.0
            if board.has_queenside_castling_rights(them):
                tensor[b_idx, 15, :, :] = 1.0

            if board.ep_square is not None:
                row = 7 - (board.ep_square // 8)
                col = board.ep_square % 8

                if flip:
                    row = 7 - row
                    col = 7 - col

                tensor[b_idx, 16, row, col] = 1.0
            tensor[b_idx, 17, :, :] = 1.0 if board.turn == chess.WHITE else -1.0

        return torch.from_numpy(tensor)

    def _square_to_coords(self, square: int, flip: bool) -> tuple[int, int]:
        """
        Convert a square index (0-63) to (row, col) coordinates.

        Args:
            square: Square index (0-63, a1=0, h8=63).
            flip: Whether to flip the board for black's perspective.

        Returns:
            (row, col) tuple where row 0 is rank 8 and col 0 is file a.
        """
        row = 7 - (square // 8)  # Convert rank to row (0 = top = rank 8)
        col = square % 8  # File to column

        if flip:
            row = 7 - row
            col = 7 - col

        return row, col

    def decode_piece_planes(self, tensor: torch.Tensor) -> dict:
        """
        Decode piece positions from a tensor (for debugging).

        Args:
            tensor: Encoded tensor of shape (18, 8, 8).

        Returns:
            Dictionary mapping piece names to lists of squares.
        """
        piece_names = [
            "our_pawns",
            "our_knights",
            "our_bishops",
            "our_rooks",
            "our_queens",
            "our_king",
            "their_pawns",
            "their_knights",
            "their_bishops",
            "their_rooks",
            "their_queens",
            "their_king",
        ]

        result = {}
        for i, name in enumerate(piece_names):
            squares = []
            plane = tensor[i].numpy() if isinstance(tensor, torch.Tensor) else tensor[i]
            for row in range(8):
                for col in range(8):
                    if plane[row, col] > 0.5:
                        # Convert back to square notation
                        rank = 8 - row
                        file = chr(ord("a") + col)
                        squares.append(f"{file}{rank}")
            result[name] = squares

        return result
