"""
Random agent for baseline comparison.
"""
import random
from typing import Optional

import chess

from .base import ChessAgent


class RandomAgent(ChessAgent):
    """
    Agent that plays uniformly random legal moves.
    
    Useful as a baseline for comparison and for generating
    diverse training data.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        self._rng = random.Random(seed)
        self._seed = seed
    
    @property
    def name(self) -> str:
        return "RandomAgent"
    
    def get_move(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
    ) -> chess.Move:
        """
        Get a random legal move.
        
        Args:
            board: Current chess board.
            time_limit: Ignored for random agent.
        
        Returns:
            Random legal move.
        """
        legal_moves = list(board.legal_moves)
        return self._rng.choice(legal_moves)
    
    def get_move_distribution(
        self,
        board: chess.Board,
        num_moves: int = 5,
        depth: Optional[int] = None,
    ) -> list[dict]:
        """
        Get random moves with equal (zero) scores.
        """
        legal_moves = list(board.legal_moves)
        selected = self._rng.sample(legal_moves, min(num_moves, len(legal_moves)))
        return [{'move': m, 'score': 0} for m in selected]
    
    def reset(self) -> None:
        """Reset the random generator with the original seed."""
        if self._seed is not None:
            self._rng = random.Random(self._seed)
