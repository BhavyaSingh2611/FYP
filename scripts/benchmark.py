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

from src.agents.learning_agent import LearningAgent
from src.agents.uci_agent import UCIAgent
from src.config import settings
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model

LOGGER = logging.getLogger(__name__)

EVAL_DEPTH = 18

DIFFICULTY_LEVELS = [
    {"name": "Novice-1320", "elo": 1320},
    {"name": "Club-1800", "elo": 1800},
    {"name": "Expert-2300", "elo": 2300},
    {"name": "IM-2800", "elo": 2800},
    {"name": "Full-3200", "elo": 3200},
]

MATE_SCORE_BASE = 10000
MATE_SCORE_STEP = 10

_log_lock = threading.Lock()


def get_outcome(result: str, model_is_white: bool) -> str:
    """Return win/loss/draw from the model's perspective."""
    if result == "1-0":
        return "win" if model_is_white else "loss"

    if result == "0-1":
        return "loss" if model_is_white else "win"

    return "draw"


def load_agent(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> LearningAgent:
    """Load model checkpoint and return a deterministic learning agent."""
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
    clean_state = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()

    encoder = get_encoder_for_model(model_name)()

    return LearningAgent(
        model=model,
        encoder=encoder,
        device=device,
        temperature=0.0,
    )  # type: ignore[arg-type]


def get_eval_cp(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = EVAL_DEPTH,
) -> int:
    """Return centipawn-style evaluation from White's perspective."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()  # type: ignore[assignment]

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


def get_game_result(
    board: chess.Board,
    move_count: int,
    max_moves: int,
) -> tuple[str, str]:
    """Return game result and termination reason."""
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


def run_game(
    white_agent: Any,
    black_agent: Any,
    engine: chess.engine.SimpleEngine,
    max_moves: int = 200,
) -> dict[str, Any]:
    """Play one game and return move-by-move metadata."""
    board = chess.Board()
    game: dict[str, Any] = {
        "moves": [],
        "san_moves": [],
        "evaluations": [get_eval_cp(engine, board)],
        "white_agent": getattr(white_agent, "name", str(white_agent)),
        "black_agent": getattr(black_agent, "name", str(black_agent)),
    }

    while not board.is_game_over() and len(game["moves"]) < max_moves:
        active = white_agent if board.turn == chess.WHITE else black_agent
        move = active.get_move(board)

        game["san_moves"].append(board.san(move))
        game["moves"].append(move.uci())

        board.push(move)
        game["evaluations"].append(get_eval_cp(engine, board))

    result, termination = get_game_result(board, len(game["moves"]), max_moves)
    game.update(
        {
            "result": result,
            "termination": termination,
            "total_moves": len(game["moves"]),
            "final_fen": board.fen(),
        }
    )
    return game


def calc_acpl(game: dict[str, Any], model_color: str) -> float:
    """Return average centipawn loss on model moves."""
    evals = game["evaluations"]
    losses: list[int] = []

    for i in range(len(evals) - 1):
        is_white_move = i % 2 == 0
        model_to_move = (model_color == "white" and is_white_move) or (model_color == "black" and not is_white_move)

        if not model_to_move:
            continue

        eval_before = evals[i]
        eval_after = evals[i + 1]
        cp_loss = eval_before - eval_after if model_color == "white" else eval_after - eval_before
        losses.append(max(cp_loss, 0))

    return sum(losses) / len(losses) if losses else 0.0


def build_pgn(game: dict[str, Any], event_name: str) -> chess.pgn.Game:
    """Create PGN with engine evaluations in comments."""
    pgn = chess.pgn.Game()
    pgn.headers.update(
        {
            "Event": event_name,
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "White": game["white_agent"],
            "Black": game["black_agent"],
            "Result": game["result"],
            "Termination": game.get("termination", "unknown"),
        }
    )

    node: GameNode = pgn

    for move_index, uci_move in enumerate(game["moves"]):
        move = chess.Move.from_uci(uci_move)
        node = node.add_variation(move)

        eval_index = move_index + 1
        if eval_index >= len(game["evaluations"]):
            continue

        evaluation = game["evaluations"][eval_index]
        if abs(evaluation) >= MATE_SCORE_BASE:
            mate_in = (MATE_SCORE_BASE - abs(evaluation)) // MATE_SCORE_STEP
            node.comment = f"[%eval #{mate_in if evaluation > 0 else -mate_in}]"
        else:
            node.comment = f"[%eval {evaluation / 100:.2f}]"

    return pgn


def _run_level(
    agent: LearningAgent,
    model_name: str,
    level: dict[str, Any],
    games_per_level: int,
    stockfish_path: str,
) -> tuple[str, dict[str, Any]]:
    level_name = level["name"]
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    opponent = UCIAgent(stockfish_path, uci_elo=level["elo"])

    wins = draws = losses = 0
    acpl_values: list[float] = []
    game_lengths: list[int] = []
    game_records: list[dict[str, Any]] = []

    try:
        for game_index in range(games_per_level):
            model_color = "white" if game_index % 2 == 0 else "black"

            game = run_game(agent, opponent, engine) if model_color == "white" else run_game(opponent, agent, engine)

            game["model_color"] = model_color
            outcome = get_outcome(game["result"], model_color == "white")

            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                draws += 1

            acpl_values.append(calc_acpl(game, model_color))
            game_lengths.append(game["total_moves"])

            with _log_lock:
                LOGGER.info(
                    "[%s] vs %s  Game %d/%d: %s (%s, %d moves, %s)",
                    model_name,
                    level_name,
                    game_index + 1,
                    games_per_level,
                    game["result"],
                    outcome[0].upper(),
                    game["total_moves"],
                    game["termination"],
                )

            pgn = build_pgn(game, f"{model_name} vs SF {level_name}")
            game_records.append(
                {
                    "pgn": str(pgn),
                    "result": game["result"],
                    "termination": game["termination"],
                    "total_moves": game["total_moves"],
                    "evaluations": game["evaluations"],
                    "model_color": model_color,
                    "outcome": outcome,
                    "white": game["white_agent"],
                    "black": game["black_agent"],
                }
            )
    finally:
        opponent.close()
        engine.quit()

    total_games = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / total_games * 100 if total_games else 0
    avg_acpl = sum(acpl_values) / len(acpl_values) if acpl_values else 0
    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    level_data = {
        "elo": level["elo"],
        "summary": {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_pct": score_pct,
            "avg_acpl": avg_acpl,
            "avg_game_length": avg_game_length,
            "acpl_list": acpl_values,
            "game_lengths": game_lengths,
        },
        "games": game_records,
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
            avg_acpl,
        )

    return level_name, level_data


def order_levels(data: dict[str, Any], levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Return level-ordered dict based on configured difficulty order."""
    return {str(level["name"]): data[level["name"]] for level in levels if level["name"] in data}


def run_benchmark(
    backbone: str,
    checkpoint_path: Path,
    stockfish_path: str,
    games_per_level: int,
    output_dir: Path,
    levels: list[dict[str, Any]] | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Run model-vs-Stockfish benchmark and write artifacts + a single JSON file."""
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_levels = levels or DIFFICULTY_LEVELS
    model_names = [backbone]
    total_games = len(model_names) * len(selected_levels) * games_per_level

    LOGGER.info(
        """
Models: %s
Difficulty levels: %d
Games per model per level: %d
Total games: %d
Worker threads: %d
    """,
        ", ".join(model_names),
        len(selected_levels),
        games_per_level,
        total_games,
        workers,
    )

    results: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "backbone": backbone,
            "checkpoint": str(checkpoint_path),
            "stockfish": stockfish_path,
            "games_per_level": games_per_level,
            "workers": workers,
            "levels": selected_levels,
            "device": str(device),
        },
        "models": {},
    }

    for model_name in model_names:
        LOGGER.info(
            "Model: %s  (%d levels in parallel)",
            model_name,
            workers,
        )

        try:
            agent = load_agent(model_name, checkpoint_path, device)
        except FileNotFoundError as exc:
            LOGGER.warning("SKIP - %s", exc)
            continue

        results["models"][model_name] = {"levels": {}}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_by_level = {
                pool.submit(
                    _run_level,
                    agent,
                    model_name,
                    level,
                    games_per_level,
                    stockfish_path,
                ): level
                for level in selected_levels
            }

            for future in as_completed(future_by_level):
                level_name, level_data = future.result()
                results["models"][model_name]["levels"][level_name] = level_data

        ordered_levels = order_levels(results["models"][model_name]["levels"], selected_levels)
        results["models"][model_name]["levels"] = ordered_levels

    json_output_path = output_dir / "benchmark_results.json"
    with open(json_output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    LOGGER.info("All outputs saved to: %s", output_dir)
    return results


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

    run_benchmark(
        backbone=args.backbone,
        checkpoint_path=Path(args.weights),
        stockfish_path=args.stockfish,
        games_per_level=args.games,
        output_dir=output_dir,
        levels=selected_levels,
        workers=args.workers,
    )

    LOGGER.info("Final artifacts in: %s", output_dir)


if __name__ == "__main__":
    main()
