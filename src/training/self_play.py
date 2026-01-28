"""
Self-play data generation for reinforcement learning.

Generates training data by having the model play against itself using MCTS.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random

import chess
import torch
import numpy as np
from tqdm import tqdm

from ..agents.mcts_agent import MCTSAgent
from ..models.base import ChessModel
from ..chess_env.board_wrapper import UCI_MOVE_TO_INDEX


@dataclass
class SelfPlayExample:
    """Single training example from self-play."""
    fen: str
    policy: Dict[str, float]  # Move UCI -> visit proportion
    value: float  # Game outcome from this position's perspective


@dataclass
class SelfPlayGame:
    """Complete self-play game."""
    examples: List[SelfPlayExample] = field(default_factory=list)
    result: str = "*"
    moves: List[str] = field(default_factory=list)
    
    @property
    def num_positions(self) -> int:
        return len(self.examples)


class SelfPlayGenerator:
    """
    Generates self-play games for training.
    
    Uses MCTS to play games, recording:
    - Position (FEN)
    - MCTS policy (visit count distribution)
    - Game outcome (from position's perspective)
    """
    
    def __init__(
        self,
        model: ChessModel,
        encoder,
        device: torch.device,
        num_simulations: int = 100,
        c_puct: float = 1.4,
        temperature_moves: int = 30,  # Use temperature for first N moves
        temperature: float = 1.0,
        dirichlet_noise: bool = True,
    ):
        """
        Initialize self-play generator.
        
        Args:
            model: Policy/value network.
            encoder: Board encoder.
            device: Torch device.
            num_simulations: MCTS simulations per move.
            c_puct: Exploration constant.
            temperature_moves: Number of moves to use temperature sampling.
            temperature: Temperature for move selection.
            dirichlet_noise: Add exploration noise at root.
        """
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature_moves = temperature_moves
        self.temperature = temperature
        self.dirichlet_noise = dirichlet_noise
    
    def generate_game(self, max_moves: int = 200) -> SelfPlayGame:
        """
        Generate a single self-play game.
        
        Returns:
            SelfPlayGame with examples and result.
        """
        game = SelfPlayGame()
        board = chess.Board()
        
        # Create MCTS agent
        agent = MCTSAgent(
            model=self.model,
            encoder=self.encoder,
            device=self.device,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            temperature=self.temperature if len(game.moves) < self.temperature_moves else 0.0,
            add_noise=self.dirichlet_noise,
        )
        
        while not board.is_game_over() and len(game.moves) < max_moves:
            # Update temperature based on move number
            agent.temperature = self.temperature if len(game.moves) < self.temperature_moves else 0.0
            
            # Get move and policy from MCTS
            move, policy = agent.get_move_with_policy(board)
            
            # Store example (value will be filled in after game ends)
            example = SelfPlayExample(
                fen=board.fen(),
                policy={m.uci(): p for m, p in policy.items()},
                value=0.0,  # Placeholder
            )
            game.examples.append(example)
            
            # Make move
            game.moves.append(move.uci())
            board.push(move)
        
        # Determine game result
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                game.result = "0-1"  # Black wins
            else:
                game.result = "1-0"  # White wins
        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.is_fifty_moves() or board.is_repetition() or \
             len(game.moves) >= max_moves:
            game.result = "1/2-1/2"
        else:
            game.result = "*"
        
        # Fill in values based on game result
        self._assign_values(game)
        
        return game
    
    def _assign_values(self, game: SelfPlayGame) -> None:
        """Assign values to examples based on game outcome."""
        if game.result == "1-0":
            final_value = 1.0  # White won
        elif game.result == "0-1":
            final_value = -1.0  # Black won
        else:
            final_value = 0.0  # Draw
        
        # Assign values - alternating perspective
        for i, example in enumerate(game.examples):
            # Even moves are white's turn, odd are black's
            if i % 2 == 0:  # White to move
                example.value = final_value
            else:  # Black to move
                example.value = -final_value
    
    def generate_games(
        self,
        num_games: int,
        max_moves: int = 200,
        show_progress: bool = True,
    ) -> List[SelfPlayGame]:
        """
        Generate multiple self-play games.
        
        Args:
            num_games: Number of games to generate.
            max_moves: Maximum moves per game.
            show_progress: Show progress bar.
        
        Returns:
            List of SelfPlayGame objects.
        """
        games = []
        
        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc="Self-play games")
        
        for _ in iterator:
            game = self.generate_game(max_moves)
            games.append(game)
        
        return games


def games_to_tensors(
    games: List[SelfPlayGame],
    encoder,
    model_type: str = "cnn",
) -> Dict[str, torch.Tensor]:
    """
    Convert self-play games to training tensors.
    
    Args:
        games: List of self-play games.
        encoder: Board encoder.
        model_type: "cnn" for tensor input, "transformer" for dict input.
    
    Returns:
        Dictionary with input tensors, 'policies', 'values'.
    """
    all_policies = []
    all_values = []
    is_dict_input = model_type in ["transformer", "square_transformer", "piece_transformer"]
    
    # For CNN: list of tensors
    all_inputs = []
    
    # For transformer: lists of each component
    all_tokens = []
    all_positions = []
    all_attention_masks = []
    all_side_to_move = []
    all_castling = []
    
    for game in games:
        for example in game.examples:
            # Encode position
            board = chess.Board(example.fen)
            encoded = encoder.encode(board)
            
            if isinstance(encoded, dict):
                # Transformer encoder - store all components
                all_tokens.append(encoded['tokens'])
                all_positions.append(encoded['positions'])
                all_attention_masks.append(encoded['attention_mask'])
                all_side_to_move.append(encoded['side_to_move'])
                all_castling.append(encoded['castling'])
            else:
                # CNN encoder - store tensor
                all_inputs.append(encoded)
            
            # Build policy tensor
            policy = torch.zeros(len(UCI_MOVE_TO_INDEX))
            for uci, prob in example.policy.items():
                idx = UCI_MOVE_TO_INDEX.get(uci, -1)
                if idx >= 0:
                    policy[idx] = prob
            all_policies.append(policy)
            
            # Value
            all_values.append(example.value)
    
    result = {
        'policies': torch.stack(all_policies),
        'values': torch.tensor(all_values, dtype=torch.float32),
        'is_dict_input': is_dict_input,
    }
    
    if is_dict_input and all_tokens:
        result['tokens'] = torch.stack(all_tokens)
        result['positions'] = torch.stack(all_positions)
        result['attention_mask'] = torch.stack(all_attention_masks)
        result['side_to_move'] = torch.stack(all_side_to_move)
        result['castling'] = torch.stack(all_castling)
    elif all_inputs:
        result['inputs'] = torch.stack(all_inputs)
    
    return result


