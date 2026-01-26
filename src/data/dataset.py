"""
PyTorch Dataset for loading chess training data from SQLite.
"""
from typing import Optional, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import chess

from .database import ChessDatabase
from ..chess_env.board_wrapper import UCI_MOVE_TO_INDEX, NUM_MOVES


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess training data.
    
    Loads positions from SQLite database and encodes them on-the-fly
    using the provided encoder.
    
    Supports:
        - Policy targets (move indices or soft distributions)
        - Value targets (game outcome from perspective of side to move)
    """
    
    def __init__(
        self,
        db_path: str | Path,
        encoder: Callable,
        use_soft_labels: bool = True,
        include_value: bool = True,
        cache_positions: bool = True,
        use_game_outcome: bool = True,
        blend_value: float = 0.0,
    ):
        """
        Initialize dataset.
        
        Args:
            db_path: Path to SQLite database.
            encoder: State encoder instance (CNNEncoder, etc.).
            use_soft_labels: If True, return soft policy targets from move distribution.
            include_value: If True, include value targets.
            cache_positions: If True, load all positions into memory.
            use_game_outcome: If True, use actual game results for value targets.
                              If False, use centipawn scores as approximation.
            blend_value: Blend factor for combining game outcome with centipawn score.
                        0.0 = pure game outcome, 1.0 = pure centipawn score.
                        Values in between create a weighted blend.
        """
        self.db_path = Path(db_path)
        self.encoder = encoder
        self.use_soft_labels = use_soft_labels
        self.include_value = include_value
        self.use_game_outcome = use_game_outcome
        self.blend_value = blend_value
        
        # Load positions (with game outcomes if needed)
        with ChessDatabase(self.db_path) as db:
            if cache_positions:
                if use_game_outcome:
                    self.positions = db.get_positions_with_outcomes()
                else:
                    self.positions = db.get_positions_with_distributions()
            else:
                # Just get count for lazy loading
                self._position_count = db.get_position_count()
                self.positions = None
    
    def __len__(self) -> int:
        if self.positions is not None:
            return len(self.positions)
        return self._position_count
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training example.
        
        Returns:
            Dictionary with:
                - 'input': Encoded board state
                - 'policy_target': Move target (index or distribution)
                - 'value_target': Game outcome (optional)
        """
        # Get position data
        if self.positions is not None:
            pos = self.positions[idx]
        else:
            with ChessDatabase(self.db_path) as db:
                positions = db.get_positions_with_distributions(limit=1, offset=idx)
                pos = positions[0]
        
        # Create board from FEN
        board = chess.Board(pos['fen'])
        
        # Encode board
        encoded = self.encoder.encode(board)
        
        # Create policy target
        if self.use_soft_labels and 'move_distribution' in pos and pos['move_distribution']:
            # Soft labels from move distribution
            policy_target = self._create_soft_policy(pos['move_distribution'])
        else:
            # Hard label (one-hot)
            move_idx = UCI_MOVE_TO_INDEX.get(pos['best_move_uci'], 0)
            policy_target = torch.zeros(NUM_MOVES)
            policy_target[move_idx] = 1.0
        
        result = {
            'policy_target': policy_target,
        }
        
        # Add encoded input
        if isinstance(encoded, torch.Tensor):
            result['input'] = encoded
        else:
            # For dict-based encodings (Transformer, GNN)
            result['input'] = encoded
        
        # Add value target if requested
        if self.include_value:
            value = self._compute_value_target(pos)
            result['value_target'] = torch.tensor([value], dtype=torch.float32)
        
        return result
    
    def _compute_value_target(self, pos: dict) -> float:
        """
        Compute value target from game outcome and/or centipawn score.
        
        Args:
            pos: Position dict with game_result, side_to_move, best_move_score.
        
        Returns:
            Value target in range [-1, 1].
        """
        game_value = 0.0
        cp_value = 0.0
        
        # Compute value from game outcome
        if self.use_game_outcome and pos.get('game_result') is not None:
            game_value = self._result_to_value(
                pos['game_result'], 
                pos.get('side_to_move', 0)
            )
        
        # Compute value from centipawn score
        if pos.get('best_move_score') is not None:
            score = pos['best_move_score']
            cp_value = max(-1.0, min(1.0, score / 1000.0))
        
        # Blend values if requested
        if self.use_game_outcome:
            if self.blend_value > 0 and pos.get('best_move_score') is not None:
                # Weighted blend: (1 - blend) * game_outcome + blend * cp_score
                return (1 - self.blend_value) * game_value + self.blend_value * cp_value
            return game_value
        else:
            return cp_value
    
    def _result_to_value(self, result: str, side_to_move: int) -> float:
        """
        Convert game result to value from perspective of side to move.
        
        Args:
            result: Game result ('1-0', '0-1', '1/2-1/2', '*').
            side_to_move: 0 for white, 1 for black.
        
        Returns:
            Value: +1 for win, -1 for loss, 0 for draw.
        """
        if result == '1-0':  # White wins
            return 1.0 if side_to_move == 0 else -1.0
        elif result == '0-1':  # Black wins
            return -1.0 if side_to_move == 0 else 1.0
        elif result == '1/2-1/2':  # Draw
            return 0.0
        else:  # Unknown result '*'
            return 0.0
    
    def _create_soft_policy(self, move_distribution: list[dict]) -> torch.Tensor:
        """
        Create soft policy target from move distribution.
        
        Uses softmax over centipawn scores to create probabilities.
        """
        policy = torch.zeros(NUM_MOVES)
        
        if not move_distribution:
            return policy
        
        # Collect scores
        indices = []
        scores = []
        
        for move_info in move_distribution:
            move_uci = move_info['move_uci']
            idx = UCI_MOVE_TO_INDEX.get(move_uci, -1)
            if idx >= 0:
                indices.append(idx)
                scores.append(move_info.get('score', 0))
        
        if not indices:
            # Fallback to first move
            first_move = move_distribution[0]['move_uci']
            idx = UCI_MOVE_TO_INDEX.get(first_move, 0)
            policy[idx] = 1.0
            return policy
        
        # Convert scores to probabilities using softmax
        # Scale scores to prevent numerical issues
        scores = torch.tensor(scores, dtype=torch.float32)
        scores = scores / 100.0  # Scale centipawns
        
        probs = torch.softmax(scores, dim=0)
        
        for idx, prob in zip(indices, probs):
            policy[idx] = prob
        
        return policy


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for batching chess data.
    
    Handles both tensor and dict-based inputs.
    """
    result = {}
    
    # Check input type from first example
    first_input = batch[0]['input']
    
    if isinstance(first_input, torch.Tensor):
        # CNN-style batching
        result['input'] = torch.stack([b['input'] for b in batch])
    else:
        # Dict-style batching (Transformer, GNN)
        result['input'] = {}
        for key in first_input.keys():
            values = [b['input'][key] for b in batch]
            if torch.is_tensor(values[0]):
                result['input'][key] = torch.stack(values)
            else:
                result['input'][key] = values
    
    # Stack policy targets
    result['policy_target'] = torch.stack([b['policy_target'] for b in batch])
    
    # Stack value targets if present
    if 'value_target' in batch[0]:
        result['value_target'] = torch.stack([b['value_target'] for b in batch])
    
    return result


def create_dataloader(
    db_path: str | Path,
    encoder,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    use_soft_labels: bool = True,
    include_value: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for chess training data.
    
    Args:
        db_path: Path to SQLite database.
        encoder: State encoder instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        use_soft_labels: Use soft policy labels from move distribution.
        include_value: Include value targets.
    
    Returns:
        PyTorch DataLoader.
    """
    dataset = ChessDataset(
        db_path=db_path,
        encoder=encoder,
        use_soft_labels=use_soft_labels,
        include_value=include_value,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
