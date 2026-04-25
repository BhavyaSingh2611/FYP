import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import torch
from chess.pgn import GameNode
from render_benchmark import render_artifacts_from_store

from src.agents.learning_agent import LearningAgent
from src.agents.uci_agent import UCIAgent
from src.config import settings
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model

LOGGER = logging.getLogger(__name__)

EVAL_DEPTH = 18

MODELS = [
    "convnet",
    "resnet",
    "square_transformer",
    "piece_transformer",
    "gcn",
    "gat",
]

DIFFICULTY_LEVELS = [
    {"name": "Novice-1320", "elo": 1320},
    {"name": "Casual-1500", "elo": 1500},
    {"name": "Club-1800", "elo": 1800},
    {"name": "Strong-2000", "elo": 2000},
    {"name": "Expert-2300", "elo": 2300},
    {"name": "Master-2500", "elo": 2500},
    {"name": "IM-2800", "elo": 2800},
    {"name": "GM-3100", "elo": 3100},
    {"name": "Full-3200", "elo": 3200},
]

MATE_SCORE_BASE = 10000
MATE_SCORE_STEP = 10

_log_lock = threading.Lock()


def get_model_outcome(result: str, is_white: bool) -> str:
    """Return win/loss/draw from the model's perspective."""
    if result == "1-0":
        return "win" if is_white else "loss"

    if result == "0-1":
        return "loss" if is_white else "win"

    return "draw"


def load_model_agent(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> LearningAgent:
    """Load a trained model checkpoint and return a deterministic learning agent."""
    model_config = settings.model.model_copy(update={"head": "dual"})
    model = create_model(model_name, model_config)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )  # nosec: checkpoint may contain non-tensor objects
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned_state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()

    encoder = get_encoder_for_model(model_name)()

    return LearningAgent(
        model=model,
        encoder=encoder,
        device=device,
        temperature=0.0,
    )  # type: ignore[arg-type]


def evaluate_position(
    evaluator: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = EVAL_DEPTH,
) -> int:
    """Return a centipawn-style evaluation from White's perspective."""
    try:
        analysis = evaluator.analyse(board, chess.engine.Limit(depth=depth))
        score = analysis["score"].white()  # type: ignore[assignment]

        if score.is_mate():
            mate_in = score.mate()

            if mate_in and mate_in > 0:
                return MATE_SCORE_BASE - mate_in * MATE_SCORE_STEP

            if mate_in:
                return -MATE_SCORE_BASE - mate_in * MATE_SCORE_STEP

            return 0

        return score.score() or 0  # type: ignore[union-attr]

    except Exception as exc:
        LOGGER.debug("Engine eval failed: %s", exc)
        return 0


def resolve_game_status(
    board: chess.Board,
    move_count: int,
    max_moves: int,
) -> tuple[str, str]:
    """Return game result and termination label."""

    if board.is_checkmate():
        return ("0-1", "checkmate") if board.turn == chess.WHITE else ("1-0", "checkmate")

    if board.is_stalemate():
        return "1/2-1/2", "stalemate"

    if board.is_insufficient_material():
        return "1/2-1/2", "insufficient_material"

    if board.is_fifty_moves():
        return "1/2-1/2", "fifty_moves"

    if board.is_repetition():
        return "1/2-1/2", "repetition"

    if move_count >= max_moves:
        return "1/2-1/2", "max_moves"

    return "*", "unknown"


def play_game(
    white_agent: Any,
    black_agent: Any,
    evaluator: chess.engine.SimpleEngine,
    max_moves: int = 200,
) -> dict[str, Any]:
    """Play one game between two agents and capture move-by-move metadata."""
    board = chess.Board()
    game_data: dict[str, Any] = {
        "moves": [],
        "san_moves": [],
        "evaluations": [evaluate_position(evaluator, board)],
        "white_agent": getattr(white_agent, "name", str(white_agent)),
        "black_agent": getattr(black_agent, "name", str(black_agent)),
    }

    while not board.is_game_over() and len(game_data["moves"]) < max_moves:
        active_agent = white_agent if board.turn == chess.WHITE else black_agent
        move = active_agent.get_move(board)

        game_data["san_moves"].append(board.san(move))
        game_data["moves"].append(move.uci())

        board.push(move)
        game_data["evaluations"].append(evaluate_position(evaluator, board))

    result, termination = resolve_game_status(board, len(game_data["moves"]), max_moves)
    game_data.update(
        {
            "result": result,
            "termination": termination,
            "total_moves": len(game_data["moves"]),
            "final_fen": board.fen(),
        }
    )
    return game_data


def compute_acpl(game_data: dict[str, Any], model_color: str) -> float:
    """Return average centipawn loss across the model's moves."""
    evaluations = game_data["evaluations"]
    move_losses: list[int] = []

    for index in range(len(evaluations) - 1):
        is_white_move = index % 2 == 0
        model_to_move = (model_color == "white" and is_white_move) or (model_color == "black" and not is_white_move)

        if not model_to_move:
            continue

        eval_before = evaluations[index]
        eval_after = evaluations[index + 1]

        cp_loss = eval_before - eval_after if model_color == "white" else eval_after - eval_before
        move_losses.append(max(cp_loss, 0))

    return sum(move_losses) / len(move_losses) if move_losses else 0.0


def create_pgn(game_data: dict[str, Any], event_name: str) -> chess.pgn.Game:
    """Create a PGN game with engine evaluations in comments."""
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": event_name,
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "White": game_data["white_agent"],
            "Black": game_data["black_agent"],
            "Result": game_data["result"],
            "Termination": game_data.get("termination", "unknown"),
        }
    )

    current_node: GameNode = game

    for move_index, uci_move in enumerate(game_data["moves"]):
        move = chess.Move.from_uci(uci_move)
        current_node = current_node.add_variation(move)

        next_eval_index = move_index + 1
        if next_eval_index >= len(game_data["evaluations"]):
            continue

        evaluation = game_data["evaluations"][next_eval_index]
        if abs(evaluation) >= MATE_SCORE_BASE:
            mate_in = (MATE_SCORE_BASE - abs(evaluation)) // MATE_SCORE_STEP
            current_node.comment = f"[%eval #{mate_in if evaluation > 0 else -mate_in}]"
        else:
            current_node.comment = f"[%eval {evaluation / 100:.2f}]"

    return game


def _evaluate_level(
    agent: LearningAgent,
    model_name: str,
    level: dict[str, Any],
    games_per_level: int,
    stockfish_path: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    level_name = level["name"]
    evaluator = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    opponent = UCIAgent(stockfish_path, uci_elo=level["elo"])

    wins = draws = losses = 0
    acpl_values: list[float] = []
    game_lengths: list[int] = []
    eval_trajectories: list[list[int]] = []
    game_records: list[dict[str, Any]] = []

    try:
        for game_index in range(games_per_level):
            model_color = "white" if game_index % 2 == 0 else "black"

            if model_color == "white":
                game_data = play_game(agent, opponent, evaluator)
            else:
                game_data = play_game(opponent, agent, evaluator)

            game_data["model_color"] = model_color
            outcome = get_model_outcome(game_data["result"], model_color == "white")

            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                draws += 1

            acpl_values.append(compute_acpl(game_data, model_color))
            game_lengths.append(game_data["total_moves"])
            eval_trajectories.append(game_data["evaluations"])

            with _log_lock:
                LOGGER.info(
                    "[%s] vs %s  Game %d/%d: %s (%s, %d moves, %s)",
                    model_name,
                    level_name,
                    game_index + 1,
                    games_per_level,
                    game_data["result"],
                    outcome[0].upper(),
                    game_data["total_moves"],
                    game_data["termination"],
                )

            pgn_game = create_pgn(game_data, f"{model_name} vs SF {level_name}")
            game_records.append(
                {
                    "pgn": str(pgn_game),
                    "result": game_data["result"],
                    "termination": game_data["termination"],
                    "total_moves": game_data["total_moves"],
                    "evaluations": game_data["evaluations"],
                    "model_color": model_color,
                    "outcome": outcome,
                    "white": game_data["white_agent"],
                    "black": game_data["black_agent"],
                }
            )
    finally:
        opponent.close()
        evaluator.quit()

    total_games = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / total_games * 100 if total_games else 0
    average_acpl = sum(acpl_values) / len(acpl_values) if acpl_values else 0
    average_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    level_summary = {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": score_pct,
        "avg_acpl": average_acpl,
        "acpl_list": acpl_values,
        "avg_game_length": average_game_length,
        "game_lengths": game_lengths,
        "eval_trajectories": eval_trajectories,
        "elo": level["elo"],
    }

    with _log_lock:
        LOGGER.info(
            "[%s] vs %s  => %dW / %dD / %dL  |  Score: %.0f%%  |  Avg ACPL: %.0f",
            model_name,
            level_name,
            wins,
            draws,
            losses,
            score_pct,
            average_acpl,
        )

    level_bundle = {"elo": level["elo"], "games": game_records}
    return level_name, level_summary, level_bundle


def reorder(
    data: dict[str, Any],
    levels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a new dict whose keys follow the configured level ordering."""
    return {str(level["name"]): data[level["name"]] for level in levels if level["name"] in data}


def run_evaluation(
    backbone: str,
    checkpoint_path: Path,
    stockfish_path: str,
    games_per_level: int,
    output_dir: Path,
    levels: list[dict[str, Any]] | None = None,
    workers: int = 4,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run model-vs-Stockfish benchmarks and write all output artifacts."""
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_levels = levels or DIFFICULTY_LEVELS
    model_names = [backbone]
    total_games = len(model_names) * len(selected_levels) * games_per_level

    LOGGER.info(
        """
    Device: %s
    Models: %s
    Difficulty levels: %d
    Games per model per level: %d
    Total games: %d
    Worker threads: %d
    """,
        device,
        ", ".join(model_names),
        len(selected_levels),
        games_per_level,
        total_games,
        workers,
    )

    results_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    benchmark_store: dict[str, dict[str, dict[str, Any]]] = {}

    for model_name in model_names:
        LOGGER.info(
            "=" * 70 + "\n  MODEL: %s  (%d levels in parallel)\n" + "=" * 70,
            model_name.upper(),
            workers,
        )

        try:
            agent = load_model_agent(model_name, checkpoint_path, device)
        except FileNotFoundError as exc:
            LOGGER.warning("SKIP — %s", exc)
            continue

        results_by_model[model_name] = {}
        benchmark_store[model_name] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_by_level = {
                pool.submit(
                    _evaluate_level,
                    agent,
                    model_name,
                    level,
                    games_per_level,
                    stockfish_path,
                ): level
                for level in selected_levels
            }

            for future in as_completed(future_by_level):
                level_name, level_result, level_bundle = future.result()
                results_by_model[model_name][level_name] = level_result
                benchmark_store[model_name][level_name] = {
                    "summary": level_result,
                    "elo": level_bundle["elo"],
                    "games": level_bundle["games"],
                }

        results_by_model[model_name] = reorder(
            results_by_model[model_name],
            selected_levels,
        )
        benchmark_store[model_name] = reorder(
            benchmark_store[model_name],
            selected_levels,
        )

        render_artifacts_from_store(model_name, benchmark_store[model_name], output_dir)

    benchmark_store_output_path = output_dir / "benchmark_store.json"
    with open(benchmark_store_output_path, "w", encoding="utf-8") as output_file:
        json.dump(benchmark_store, output_file, indent=2)

    json_results = {
        model_name: {
            level_name: {key: value for key, value in level_result.items() if key != "eval_trajectories"}
            for level_name, level_result in level_results.items()
        }
        for model_name, level_results in results_by_model.items()
    }

    json_output_path = output_dir / "evaluation_results.json"
    with open(json_output_path, "w", encoding="utf-8") as output_file:
        json.dump(json_results, output_file, indent=2)

    LOGGER.info("All outputs saved to: %s", output_dir)
    return benchmark_store


def parse_selected_levels(level_argument: str | None) -> list[dict[str, Any]]:
    """Return selected difficulty levels from a comma-separated index string."""
    if not level_argument:
        return DIFFICULTY_LEVELS

    indices = [int(part.strip()) for part in level_argument.split(",")]
    return [DIFFICULTY_LEVELS[index] for index in indices]


def main() -> None:
    """Parse CLI arguments and run the benchmark pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run comprehensive Elo benchmark pipeline for a chess model.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        help="Network backbone architecture",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="/opt/homebrew/bin/stockfish",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=4,
        help="Games per difficulty level (use even number)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluation",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated level indices (0-9)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel threads (each spawns its own Stockfish)",
    )

    args = parser.parse_args()
    selected_levels = parse_selected_levels(args.levels)
    output_dir = Path(args.output_dir)

    LOGGER.info("=" * 70 + "\n  COMPREHENSIVE MODEL EVALUATION\n" + "=" * 70)
    LOGGER.info("Running in-memory benchmark + final artifact export...")

    run_evaluation(
        backbone=args.backbone,
        checkpoint_path=Path(args.weights),
        stockfish_path=args.stockfish,
        games_per_level=args.games,
        output_dir=output_dir,
        levels=selected_levels,
        workers=args.workers,
    )

    LOGGER.info("=" * 70 + "\n  EVALUATION COMPLETE!\n" + "=" * 70)
    LOGGER.info("Final artifacts in: %s", output_dir)


if __name__ == "__main__":
    main()
