"""
Stockfish-based reinforcement learning with dense per-move reward signals.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import chess
import chess.engine
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..agents.uci_agent import UCIAgent
from ..models.base import ChessModel
from ..chess_env.board_wrapper import UCI_MOVE_TO_INDEX


@dataclass
class StockfishRLExample:
    fen: str
    policy: Dict[str, float]
    value: float


@dataclass
class StockfishRLGame:
    examples: List[StockfishRLExample] = field(default_factory=list)
    result: str = "*"
    moves: List[str] = field(default_factory=list)
    model_color: bool = chess.WHITE

    @property
    def num_positions(self) -> int:
        return len(self.examples)


class StockfishRLGenerator:
    """
    Generates training data by playing the model against Stockfish.

    Uses dense per-move rewards from Stockfish evaluation instead of
    sparse game-outcome rewards.
    """

    def __init__(
        self,
        model: ChessModel,
        encoder,
        device: torch.device,
        stockfish_path: str | Path,
        skill_level: int = 0,
        opponent_depth: int = 1,
        eval_depth: int = 8,
        temperature: float = 0.5,
        reward_scale: float = 1.0,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.stockfish_path = Path(stockfish_path)
        self.skill_level = skill_level
        self.opponent_depth = opponent_depth
        self.eval_depth = eval_depth
        self.temperature = temperature
        self.reward_scale = reward_scale

        self.model.eval()

    def _eval_position(self, engine: chess.engine.SimpleEngine, board: chess.Board) -> float:
        info = engine.analyse(board, chess.engine.Limit(depth=self.eval_depth))
        score = info["score"].white()
        if score.is_mate():
            mate_moves = score.mate()
            return 10000.0 if mate_moves > 0 else -10000.0
        return float(score.score())

    @torch.no_grad()
    def _get_model_policy_and_move(self, board: chess.Board) -> tuple[Dict[str, float], chess.Move]:
        encoded = self.encoder.encode(board)

        if isinstance(encoded, torch.Tensor):
            x = encoded.unsqueeze(0).to(self.device)
        elif isinstance(encoded, dict):
            x = {}
            for k, v in encoded.items():
                if not torch.is_tensor(v):
                    x[k] = v
                elif k in ('edge_index', 'edge_attr'):
                    x[k] = v.to(self.device)
                else:
                    x[k] = v.unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unknown encoded type: {type(encoded)}")

        output = self.model(x)
        policy_logits = output['policy'][0]

        legal_moves = list(board.legal_moves)
        legal_indices = []
        legal_move_list = []
        for m in legal_moves:
            idx = UCI_MOVE_TO_INDEX.get(m.uci(), -1)
            if idx >= 0:
                legal_indices.append(idx)
                legal_move_list.append(m)

        if not legal_indices:
            move = legal_moves[0]
            return {move.uci(): 1.0}, move

        mask = torch.full_like(policy_logits, float('-inf'))
        mask[legal_indices] = 0
        masked_logits = policy_logits + mask

        probs = F.softmax(masked_logits / self.temperature, dim=-1)

        policy = {}
        for m, idx in zip(legal_move_list, legal_indices):
            policy[m.uci()] = probs[idx].item()

        move_idx = torch.multinomial(probs, 1).item()

        try:
            move_uci = None
            for m, idx in zip(legal_move_list, legal_indices):
                if idx == move_idx:
                    move_uci = m
                    break
            if move_uci is None:
                move_uci = legal_move_list[0]
        except (ValueError, IndexError):
            move_uci = legal_move_list[0]

        return policy, move_uci

    def generate_game(self, max_moves: int = 200) -> StockfishRLGame:
        model_color = random.choice([chess.WHITE, chess.BLACK])
        game = StockfishRLGame(model_color=model_color)
        board = chess.Board()

        opponent = UCIAgent(
            engine_path=self.stockfish_path,
            depth=self.opponent_depth,
            skill_level=self.skill_level,
        )
        eval_engine = chess.engine.SimpleEngine.popen_uci(str(self.stockfish_path))

        try:
            eval_before = self._eval_position(eval_engine, board)

            while not board.is_game_over() and len(game.moves) < max_moves:
                if board.turn == model_color:
                    fen_before = board.fen()
                    policy, move = self._get_model_policy_and_move(board)

                    board.push(move)
                    game.moves.append(move.uci())

                    eval_after = self._eval_position(eval_engine, board)

                    if model_color == chess.WHITE:
                        delta = eval_after - eval_before
                    else:
                        delta = eval_before - eval_after

                    delta = max(-1500.0, min(1500.0, delta))
                    dense_reward = delta / (1500.0 * self.reward_scale)

                    game.examples.append(StockfishRLExample(
                        fen=fen_before,
                        policy=policy,
                        value=dense_reward,
                    ))

                    eval_before = eval_after
                else:
                    move = opponent.get_move(board)
                    board.push(move)
                    game.moves.append(move.uci())
                    eval_before = self._eval_position(eval_engine, board)

            if board.is_checkmate():
                game.result = "0-1" if board.turn == chess.WHITE else "1-0"
            elif board.is_stalemate() or board.is_insufficient_material() or \
                 board.is_fifty_moves() or board.is_repetition() or \
                 len(game.moves) >= max_moves:
                game.result = "1/2-1/2"
            else:
                game.result = "*"

            self._blend_game_outcome(game)

        finally:
            opponent.close()
            eval_engine.quit()

        return game

    def _blend_game_outcome(self, game: StockfishRLGame) -> None:
        if game.result == "1-0":
            outcome = 1.0 if game.model_color == chess.WHITE else -1.0
        elif game.result == "0-1":
            outcome = 1.0 if game.model_color == chess.BLACK else -1.0
        else:
            outcome = 0.0

        for example in game.examples:
            example.value = 0.7 * example.value + 0.3 * outcome

    def generate_games(
        self,
        num_games: int,
        max_moves: int = 200,
        show_progress: bool = True,
    ) -> List[StockfishRLGame]:
        games = []

        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc="Stockfish RL games")

        for _ in iterator:
            game = self.generate_game(max_moves)
            games.append(game)

        return games


def stockfish_games_to_tensors(
    games: List[StockfishRLGame],
    encoder,
    model_type: str = "cnn",
) -> Dict[str, torch.Tensor]:
    all_policies = []
    all_values = []
    is_dict_input = model_type in ["transformer", "square_transformer", "piece_transformer"]
    is_gnn_input = model_type in ["gcn", "gat"]

    all_inputs = []
    all_tokens = []
    all_positions = []
    all_attention_masks = []
    all_side_to_move = []
    all_castling = []

    all_node_features = []
    all_edge_indices = []
    all_edge_attrs = []
    all_gnn_side_to_move = []
    all_gnn_castling = []

    for game in games:
        for example in game.examples:
            board = chess.Board(example.fen)
            encoded = encoder.encode(board)

            if is_gnn_input and isinstance(encoded, dict) and 'edge_index' in encoded:
                all_node_features.append(encoded['x'])
                all_edge_indices.append(encoded['edge_index'])
                if 'edge_attr' in encoded:
                    all_edge_attrs.append(encoded['edge_attr'])
                all_gnn_side_to_move.append(encoded['side_to_move'])
                all_gnn_castling.append(encoded['castling'])
            elif isinstance(encoded, dict):
                all_tokens.append(encoded['tokens'])
                all_positions.append(encoded['positions'])
                all_attention_masks.append(encoded['attention_mask'])
                all_side_to_move.append(encoded['side_to_move'])
                all_castling.append(encoded['castling'])
            else:
                all_inputs.append(encoded)

            policy = torch.zeros(len(UCI_MOVE_TO_INDEX))
            for uci, prob in example.policy.items():
                idx = UCI_MOVE_TO_INDEX.get(uci, -1)
                if idx >= 0:
                    policy[idx] = prob
            all_policies.append(policy)

            all_values.append(example.value)

    result = {
        'policies': torch.stack(all_policies),
        'values': torch.tensor(all_values, dtype=torch.float32),
        'is_dict_input': is_dict_input,
        'is_gnn_input': is_gnn_input,
    }

    if is_gnn_input and all_node_features:
        result['node_features'] = torch.stack(all_node_features)
        result['edge_indices'] = all_edge_indices
        result['edge_attrs'] = all_edge_attrs if all_edge_attrs else None
        result['side_to_move'] = torch.stack(all_gnn_side_to_move)
        result['castling'] = torch.stack(all_gnn_castling)
    elif is_dict_input and all_tokens:
        result['tokens'] = torch.stack(all_tokens)
        result['positions'] = torch.stack(all_positions)
        result['attention_mask'] = torch.stack(all_attention_masks)
        result['side_to_move'] = torch.stack(all_side_to_move)
        result['castling'] = torch.stack(all_castling)
    elif all_inputs:
        result['inputs'] = torch.stack(all_inputs)

    return result
