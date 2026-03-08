"""
Abstract base class for state encoders.
"""

from abc import ABC, abstractmethod
from typing import Any

import chess


class StateEncoder(ABC):
    """
    Abstract base class for chess board state encoders.

    Each encoder converts a chess.Board into a format suitable for
    a specific neural network architecture.
    """

    @abstractmethod
    def encode(self, board: chess.Board) -> Any:
        """
        Encode a chess board state.

        Args:
            board: A python-chess Board object.

        Returns:
            Encoded representation suitable for the target architecture.
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple:
        """
        Get the shape of the encoded input.

        Returns:
            Tuple describing the input shape.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the encoder name."""
        pass
