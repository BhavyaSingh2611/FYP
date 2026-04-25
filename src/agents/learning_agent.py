"""
Learning agent that wraps PyTorch models for move prediction.
"""

import chess
import torch
import torch.nn.functional as F

from ..chess_env.encoders.base import StateEncoder
from ..chess_env.move_index import INDEX_TO_UCI_MOVE, UCI_MOVE_TO_INDEX
from ..models.base import ChessModel
from .base import ChessAgent


class LearningAgent(ChessAgent):
    """
    Agent that uses a trained PyTorch model for move selection.

    Supports:
        - Policy-only models: Softmax sampling or argmax
        - Value-only models: Greedy search over legal moves (slow)
        - Dual-headed models: Policy with optional value filtering
    """

    def __init__(
        self,
        model: ChessModel,
        encoder: StateEncoder,
        device: torch.device,
        temperature: float = 0.0,
        top_k: int = 0,
        agent_name: str | None = None,
    ):
        """
        Initialize learning agent.

        Args:
            model: Trained PyTorch model.
            encoder: State encoder matching the model type.
            device: Device to run inference on.
            temperature: Sampling temperature (0 = greedy).
            top_k: If > 0, sample from top-k moves only.
            agent_name: Optional custom name for the agent.
        """
        self.model = model
        self.encoder = encoder
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self._name = agent_name or f"{model.name}"

        # Set model to eval mode
        self.model.eval()

    @property
    def name(self) -> str:
        return self._name

    def _prepare_encoded(self, encoded: torch.Tensor | dict) -> torch.Tensor | dict:
        """Move encoded board state to device, handling tensor and graph dict inputs."""
        if isinstance(encoded, torch.Tensor):
            return encoded.unsqueeze(0).to(self.device)

        x: dict = {}
        for k, v in encoded.items():
            if not torch.is_tensor(v):
                x[k] = v
            elif k in ("edge_index", "edge_attr"):
                x[k] = v.to(self.device)
            else:
                x[k] = v.unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def get_move(
        self,
        board: chess.Board,
        time_limit: float | None = None,
    ) -> chess.Move:
        """
        Get the best move using the model's policy head.

        Args:
            board: Current chess board.
            time_limit: Ignored for neural network agent.

        Returns:
            Selected move.
        """
        # Encode the board
        encoded = self.encoder.encode(board)
        x = self._prepare_encoded(encoded)
        if not isinstance(encoded, (torch.Tensor, dict)):
            raise ValueError(f"Unknown encoded type: {type(encoded)}")

        # Forward pass
        output = self.model(x)

        if "policy" not in output:
            raise ValueError("Model must have a policy head for move selection")

        # Get policy logits
        policy_logits = output["policy"][0]  # (NUM_MOVES,)

        # Mask illegal moves
        legal_moves = list(board.legal_moves)
        legal_indices = [UCI_MOVE_TO_INDEX.get(m.uci(), -1) for m in legal_moves]
        legal_indices = [i for i in legal_indices if i >= 0]

        if not legal_indices:
            # Fallback to first legal move if no moves in index
            return legal_moves[0]

        # Create mask for legal moves
        mask = torch.full_like(policy_logits, float("-inf"))
        mask[legal_indices] = 0
        masked_logits = policy_logits + mask

        # Select move
        if self.temperature == 0:
            # Greedy selection
            move_idx = masked_logits.argmax().item()
        else:
            # Temperature sampling
            probs = F.softmax(masked_logits / self.temperature, dim=-1)

            if self.top_k > 0:
                # Top-k sampling
                top_probs, top_indices = probs.topk(min(self.top_k, len(legal_indices)))
                selected = torch.multinomial(top_probs, 1)
                move_idx = top_indices[selected].item()
            else:
                move_idx = torch.multinomial(probs, 1).item()

        # Convert index to move
        move_uci = INDEX_TO_UCI_MOVE.get(move_idx)  # type: ignore
        if move_uci is None:
            return legal_moves[0]

        try:
            return chess.Move.from_uci(move_uci)
        except ValueError:
            return legal_moves[0]

    @torch.no_grad()
    def get_move_with_info(
        self,
        board: chess.Board,
        time_limit: float | None = None,
    ) -> dict:
        """
        Get move with policy probabilities and optional value.
        """
        # Encode and forward
        encoded = self.encoder.encode(board)
        x = self._prepare_encoded(encoded)

        output = self.model(x)

        move = self.get_move(board, time_limit)

        result = {"move": move}

        if "value" in output:
            result["value"] = output["value"][0].item()

        if "policy" in output:
            move_idx = UCI_MOVE_TO_INDEX.get(move.uci(), -1)
            if move_idx >= 0:
                probs = F.softmax(output["policy"][0], dim=-1)
                result["probability"] = probs[move_idx].item()  # type: ignore

        return result

    @torch.no_grad()
    def get_move_distribution(
        self,
        board: chess.Board,
        num_moves: int = 5,
        depth: int | None = None,
    ) -> list[dict]:
        """
        Get top moves with their probabilities.
        """
        # Encode and forward
        encoded = self.encoder.encode(board)
        x = self._prepare_encoded(encoded)

        output = self.model(x)

        if "policy" not in output:
            move = self.get_move(board)
            return [{"move": move, "score": 0}]

        policy_logits = output["policy"][0]

        # Get legal move indices
        legal_moves = list(board.legal_moves)
        legal_indices = [(m, UCI_MOVE_TO_INDEX.get(m.uci(), -1)) for m in legal_moves]
        legal_indices = [(m, i) for m, i in legal_indices if i >= 0]

        if not legal_indices:
            return [{"move": legal_moves[0], "score": 0}]

        # Get probabilities for legal moves
        probs = F.softmax(policy_logits, dim=-1)
        move_probs = [(m, probs[i].item()) for m, i in legal_indices]
        move_probs.sort(key=lambda x: x[1], reverse=True)

        # Return top moves
        results = []
        for move, prob in move_probs[:num_moves]:
            # Convert probability to pseudo-centipawn score
            # Higher prob = higher score
            score = int(prob * 1000)  # Scale to centipawns
            results.append({"move": move, "score": score, "probability": prob})

        return results

    def reset(self) -> None:
        """Reset the agent's internal state (if any)."""
        pass

    def close(self) -> None:
        """Clean up resources (if any)."""
        pass
