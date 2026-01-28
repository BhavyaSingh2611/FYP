"""
Monte Carlo Tree Search (MCTS) Agent for chess.

Implements AlphaZero-style MCTS using a policy/value neural network for
move prior estimation and position evaluation.
"""
import math
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import chess
import torch
import torch.nn.functional as F

from .base import ChessAgent
from ..models.base import ChessModel
from ..chess_env.board_wrapper import UCI_MOVE_TO_INDEX, INDEX_TO_UCI_MOVE


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree.
    
    Stores statistics for selecting promising moves during search.
    """
    move: Optional[chess.Move] = None  # Move that led to this node
    parent: Optional['MCTSNode'] = None
    
    # MCTS statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Policy network prior probability
    
    # Children
    children: Dict[chess.Move, 'MCTSNode'] = field(default_factory=dict)
    is_expanded: bool = False
    
    @property
    def value(self) -> float:
        """Average value from all visits."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Upper Confidence Bound score for node selection.
        
        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)
        
        Where:
            Q = average value
            P = prior probability
            N = visit count
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest UCB score."""
        return max(
            self.children.values(),
            key=lambda n: n.ucb_score(c_puct, self.visit_count)
        )
    
    def expand(self, move_priors: Dict[chess.Move, float]) -> None:
        """Expand node with children for all legal moves."""
        for move, prior in move_priors.items():
            self.children[move] = MCTSNode(
                move=move,
                parent=self,
                prior=prior,
            )
        self.is_expanded = True
    
    def backup(self, value: float) -> None:
        """Backpropagate value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent's perspective
            node = node.parent


class MCTSAgent(ChessAgent):
    """
    MCTS agent using neural network for move priors and position evaluation.
    
    Performs tree search to find the best move, guided by:
    - Policy network: Provides prior probabilities for move selection
    - Value network: Evaluates leaf positions
    """
    
    def __init__(
        self,
        model: ChessModel,
        encoder,
        device: torch.device,
        num_simulations: int = 100,
        c_puct: float = 1.4,
        temperature: float = 0.0,
        add_noise: bool = False,
        noise_epsilon: float = 0.25,
        noise_alpha: float = 0.3,
    ):
        """
        Initialize MCTS agent.
        
        Args:
            model: Trained policy/value network.
            encoder: Board encoder for the model.
            device: Device for inference.
            num_simulations: Number of MCTS iterations per move.
            c_puct: Exploration constant for UCB.
            temperature: Temperature for move selection (0=greedy).
            add_noise: Add Dirichlet noise to root priors (for self-play).
            noise_epsilon: Weight of noise vs prior.
            noise_alpha: Dirichlet noise concentration parameter.
        """
        self.model = model
        self.encoder = encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
        
        self._name = f"MCTS_{model.name}_{num_simulations}sims"
        self.model.eval()
    
    @property
    def name(self) -> str:
        return self._name
    
    @torch.no_grad()
    def get_move(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
    ) -> chess.Move:
        """
        Get best move using MCTS.
        
        Args:
            board: Current chess position.
            time_limit: Ignored (uses num_simulations instead).
        
        Returns:
            Best move found by MCTS.
        """
        root = self._search(board)
        return self._select_move(root)
    
    def get_move_with_policy(
        self,
        board: chess.Board,
    ) -> tuple[chess.Move, Dict[chess.Move, float]]:
        """
        Get move along with MCTS visit count policy.
        
        Returns:
            Tuple of (move, policy_dict) where policy_dict maps
            moves to their visit count proportions.
        """
        root = self._search(board)
        
        # Build policy from visit counts
        total_visits = sum(c.visit_count for c in root.children.values())
        policy = {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }
        
        move = self._select_move(root)
        return move, policy
    
    def _search(self, board: chess.Board) -> MCTSNode:
        """
        Perform MCTS from the given position.
        
        Returns:
            Root node after search.
        """
        root = MCTSNode()
        
        # Expand root with policy priors
        policy, value = self._evaluate(board)
        
        # Add Dirichlet noise for exploration during self-play
        if self.add_noise:
            noise = torch.distributions.Dirichlet(
                torch.full((len(policy),), self.noise_alpha)
            ).sample().tolist()
            
            for i, (move, prior) in enumerate(policy.items()):
                policy[move] = (1 - self.noise_epsilon) * prior + self.noise_epsilon * noise[i]
        
        root.expand(policy)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()
            
            # Selection: traverse tree to leaf
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                sim_board.push(node.move)
            
            # Check for terminal state
            if sim_board.is_game_over():
                result = sim_board.result()
                if result == "1-0":
                    value = 1.0 if sim_board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = -1.0 if sim_board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0
            else:
                # Expansion and evaluation
                policy, value = self._evaluate(sim_board)
                node.expand(policy)
            
            # Backpropagation
            node.backup(value)
        
        return root
    
    def _evaluate(self, board: chess.Board) -> tuple[Dict[chess.Move, float], float]:
        """
        Evaluate position with neural network.
        
        Returns:
            Tuple of (policy_dict, value) where policy_dict maps
            legal moves to prior probabilities.
        """
        # Encode board
        encoded = self.encoder.encode(board)
        
        if isinstance(encoded, torch.Tensor):
            x = encoded.unsqueeze(0).to(self.device)
        else:
            x = {k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v 
                 for k, v in encoded.items()}
        
        # Forward pass
        output = self.model(x)
        
        # Get policy for legal moves
        policy_logits = output['policy'][0]
        
        legal_moves = list(board.legal_moves)
        legal_indices = [UCI_MOVE_TO_INDEX.get(m.uci(), -1) for m in legal_moves]
        
        # Mask illegal moves
        mask = torch.full_like(policy_logits, float('-inf'))
        valid_moves = []
        valid_indices = []
        
        for move, idx in zip(legal_moves, legal_indices):
            if idx >= 0:
                mask[idx] = 0
                valid_moves.append(move)
                valid_indices.append(idx)
        
        masked_logits = policy_logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        
        # Build policy dictionary
        policy = {}
        for move, idx in zip(valid_moves, valid_indices):
            policy[move] = probs[idx].item()
        
        # Handle case where no valid moves are in index
        if not policy and legal_moves:
            uniform_prob = 1.0 / len(legal_moves)
            policy = {m: uniform_prob for m in legal_moves}
        
        # Get value
        value = output['value'][0].item() if 'value' in output else 0.0
        
        # Adjust value for side to move (network outputs from current player's POV)
        # No adjustment needed if value is already relative to current player
        
        return policy, value
    
    def _select_move(self, root: MCTSNode) -> chess.Move:
        """
        Select move from root based on visit counts.
        
        Uses temperature for exploration vs exploitation.
        """
        if not root.children:
            raise ValueError("No moves available")
        
        if self.temperature == 0:
            # Greedy: select most visited
            return max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            # Temperature sampling
            visits = torch.tensor([c.visit_count for c in root.children.values()], dtype=torch.float)
            
            if self.temperature == float('inf'):
                # Uniform random
                probs = torch.ones_like(visits) / len(visits)
            else:
                # Softmax with temperature on log visits
                log_visits = torch.log(visits + 1e-8)
                probs = F.softmax(log_visits / self.temperature, dim=-1)
            
            idx = torch.multinomial(probs, 1).item()
            return list(root.children.keys())[idx]
    
    def get_move_distribution(
        self,
        board: chess.Board,
        num_moves: int = 5,
        depth: Optional[int] = None,
    ) -> List[dict]:
        """
        Get top moves with visit counts from MCTS.
        """
        root = self._search(board)
        
        # Sort by visit count
        sorted_children = sorted(
            root.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True
        )[:num_moves]
        
        total_visits = sum(c.visit_count for c in root.children.values())
        
        results = []
        for move, child in sorted_children:
            results.append({
                'move': move,
                'visits': child.visit_count,
                'probability': child.visit_count / total_visits if total_visits > 0 else 0,
                'value': child.value,
                'score': int(child.value * 100),  # Pseudo-centipawns
            })
        
        return results
