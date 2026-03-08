"""
Abstract base class for chess agents.
"""

from abc import ABC, abstractmethod

import chess


class ChessAgent(ABC):
    """
    Abstract base class for all chess-playing agents.

    All agents must implement:
        - get_move(): Return the best move for the current position
        - name: Property returning the agent's name
    """

    @abstractmethod
    def get_move(
        self,
        board: chess.Board,
        time_limit: float | None = None,
    ) -> chess.Move:
        """
        Get the best move for the current position.

        Args:
            board: Current chess board state.
            time_limit: Optional time limit in seconds for move selection.

        Returns:
            Selected move.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent's name."""
        pass

    def get_move_with_info(
        self,
        board: chess.Board,
        time_limit: float | None = None,
    ) -> dict:
        """
        Get move with additional information.

        Args:
            board: Current chess board state.
            time_limit: Optional time limit.

        Returns:
            Dictionary with 'move' and optional 'score', 'pv', etc.
        """
        move = self.get_move(board, time_limit)
        return {"move": move}

    def get_move_distribution(
        self,
        board: chess.Board,
        num_moves: int = 5,
        depth: int | None = None,
    ) -> list[dict]:
        """
        Get a distribution of top moves with scores.

        Args:
            board: Current chess board state.
            num_moves: Number of top moves to return.
            depth: Search depth (if applicable).

        Returns:
            List of dicts with 'move' and 'score' (centipawns).

        Note: Base implementation returns only the best move.
              Override in subclasses for multi-move analysis.
        """
        move = self.get_move(board)
        return [{"move": move, "score": 0}]

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's internal state (if any)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (if any)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
