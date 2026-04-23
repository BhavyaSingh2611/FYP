#!/usr/bin/env python3
"""
Unified benchmark script for chess model evaluation and analysis.

Combines: evaluate_all, split_pgn, plot_individual_games, plot_eval_trajectories
into a single CLI with subcommands.

Usage:
    python scripts/benchmark.py evaluate --checkpoint-dir runs/50_10M --output-dir runs/50_10M/evaluation
    python scripts/benchmark.py split    --input all_evaluation_games.pgn --output-dir split/
    python scripts/benchmark.py plot-games       --input-dir split/ --output-dir figures/individual_games
    python scripts/benchmark.py plot-trajectories --input-dir split/ --output-dir figures/trajectories
    python scripts/benchmark.py all      --checkpoint-dir runs/50_10M --output-dir runs/50_10M/evaluation
"""

import argparse
import json
import logging
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import torch
from chess.pgn import GameNode
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

matplotlib.use("Agg")

from src.agents.learning_agent import LearningAgent
from src.agents.uci_agent import UCIAgent
from src.config import settings
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_PATTERN = re.compile(r"\[%eval ([^\]]+)\]")
EVAL_CAP = 10.0
SMOOTH_WINDOW = 5

MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]

DIFFICULTY_LEVELS = [
    {"name": "Beginner-800", "elo": 800},
    {"name": "Novice-1100", "elo": 1100},
    {"name": "Casual-1400", "elo": 1400},
    {"name": "Club-1700", "elo": 1700},
    {"name": "Strong-2000", "elo": 2000},
    {"name": "Expert-2300", "elo": 2300},
    {"name": "Master-2500", "elo": 2500},
    {"name": "IM-2800", "elo": 2800},
    {"name": "GM-3100", "elo": 3100},
    {"name": "Full-3200", "elo": 3200},
]

EVAL_DEPTH = 18

MODEL_COLORS = {
    "convnet": "#1f77b4",
    "resnet": "#ff7f0e",
    "square_transformer": "#2ca02c",
    "piece_transformer": "#d62728",
    "gcn": "#9467bd",
    "gat": "#8c564b",
}
MODEL_LABELS = {
    "convnet": "ConvNet",
    "resnet": "ResNet",
    "square_transformer": "SqTransformer",
    "piece_transformer": "PcTransformer",
    "gcn": "GCN",
    "gat": "GAT",
}

OUTCOME_COLORS = {"win": "#2ecc71", "loss": "#e74c3c", "draw": "#95a5a6"}
OUTCOME_LABELS = {"win": "Model wins", "loss": "Model loses", "draw": "Draw"}

_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Shared PGN parsing helpers
# ---------------------------------------------------------------------------


def parse_games_from_file(pgn_path: Path) -> list[str]:
    """Split a PGN file into individual game strings."""
    games: list[str] = []
    current: list[str] = []

    with open(pgn_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[Event ") and current:
                games.append("\n".join(current))
                current = []
            current.append(line.rstrip())

    if current:
        games.append("\n".join(current))
    return games


def extract_evals(move_text: str) -> list[float]:
    """Extract clamped evaluation values from a move text string."""
    raw_evals = EVAL_PATTERN.findall(move_text)
    evals: list[float] = []

    for i, raw in enumerate(raw_evals):
        is_white_move = i % 2 == 0
        if raw.startswith("#"):
            val = EVAL_CAP if is_white_move else -EVAL_CAP
        elif raw.startswith("-#"):
            val = -EVAL_CAP
        else:
            val = max(-EVAL_CAP, min(EVAL_CAP, float(raw)))
        evals.append(val)

    return evals


def get_move_text(game_text: str) -> str:
    """Return concatenated non-header lines from a PGN game string."""
    return " ".join(
        line.strip()
        for line in game_text.split("\n")
        if line.strip() and not line.strip().startswith("[")
    )


def header_value(game_text: str, tag: str) -> str | None:
    """Extract a PGN header value by tag name."""
    m = re.search(rf'\[{tag} "([^"]+)"\]', game_text)
    return m.group(1) if m else None


def parse_model_elo_from_stem(stem: str) -> tuple[str, str]:
    """Extract (model_name, elo) from a PGN filename stem like 'convnet_1400'."""
    parts = stem.rsplit("_", 1)
    model_name = parts[0].replace("_", " ").title()
    elo = parts[1] if len(parts) == 2 else "?"
    return model_name, elo


def model_outcome(result: str, model_is_white: bool) -> str:
    if result == "1-0":
        return "win" if model_is_white else "loss"
    if result == "0-1":
        return "loss" if model_is_white else "win"
    return "draw"


def smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(values) < window:
        return values

    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


# ---------------------------------------------------------------------------
# 1. Evaluate: play games against Stockfish and produce results
# ---------------------------------------------------------------------------


def load_model_agent(
    model_name: str, checkpoint_path: Path, device: torch.device
) -> LearningAgent:
    model_cfg = settings.model.model_copy(update={"head": "dual"})
    model = create_model(model_name, model_cfg)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec: checkpoint may contain non-tensor objects
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    encoder = get_encoder_for_model(model_name)()
    return LearningAgent(model=model, encoder=encoder, device=device, temperature=0.0)  # type: ignore


def evaluate_position(
    engine: chess.engine.SimpleEngine, board: chess.Board, depth: int = EVAL_DEPTH
) -> int:
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()  # type: ignore
        if score.is_mate():
            mate_in = score.mate()
            return (10000 - mate_in * 10) if mate_in > 0 else (-10000 - mate_in * 10)  # type: ignore
        return score.score()  # type: ignore
    except Exception as e:
        LOGGER.debug("Engine eval failed: %s", e)
        return 0


def play_game(white_agent, black_agent, evaluator, max_moves: int = 200) -> dict:
    board = chess.Board()
    data: dict[str, Any] = {
        "moves": [],
        "san_moves": [],
        "evaluations": [evaluate_position(evaluator, board)],
        "white_agent": getattr(white_agent, "name", str(white_agent)),
        "black_agent": getattr(black_agent, "name", str(black_agent)),
    }

    while not board.is_game_over() and len(data["moves"]) < max_moves:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        move = agent.get_move(board)
        data["san_moves"].append(board.san(move))
        data["moves"].append(move.uci())
        board.push(move)
        data["evaluations"].append(evaluate_position(evaluator, board))


    result, termination = "*", "unknown"
    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        termination = "checkmate"
    elif board.is_stalemate():
        result, termination = "1/2-1/2", "stalemate"
    elif board.is_insufficient_material():
        result, termination = "1/2-1/2", "insufficient_material"
    elif board.is_fifty_moves():
        result, termination = "1/2-1/2", "fifty_moves"
    elif board.is_repetition():
        result, termination = "1/2-1/2", "repetition"
    elif len(data["moves"]) >= max_moves:
        result, termination = "1/2-1/2", "max_moves"

    data.update(
        {
            "result": result,
            "termination": termination,
            "total_moves": len(data["moves"]),
            "final_fen": board.fen(),
        }
    )
    return data


def compute_acpl(game_data: dict, model_color: str) -> float:
    """Average centipawn loss for the model's moves."""
    evals = game_data["evaluations"]
    losses = []

    for i in range(len(evals) - 1):
        is_white_move = i % 2 == 0

        if (model_color == "white") != is_white_move:
            continue

        before, after = evals[i], evals[i + 1]
        loss = (before - after) if model_color == "white" else (after - before)
        losses.append(max(loss, 0))

    return sum(losses) / len(losses) if losses else 0.0


def create_pgn(game_data: dict, event: str) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": event,
            "Date": datetime.now().strftime("%Y.%m.%d"),
            "White": game_data["white_agent"],
            "Black": game_data["black_agent"],
            "Result": game_data["result"],
            "Termination": game_data.get("termination", "unknown"),
        }
    )

    node: GameNode = game
    board = chess.Board()
    for i, uci_move in enumerate(game_data["moves"]):
        move = chess.Move.from_uci(uci_move)
        node = node.add_variation(move)

        if i + 1 < len(game_data["evaluations"]):
            ev = game_data["evaluations"][i + 1]
            if abs(ev) >= 10000:
                mate_in = (10000 - abs(ev)) // 10
                node.comment = f"[%eval #{mate_in if ev > 0 else -mate_in}]"
            else:
                node.comment = f"[%eval {ev / 100:.2f}]"
        board.push(move)

    return game


def _evaluate_level(
    agent: LearningAgent,
    model_name: str,
    lvl: dict,
    games_per_level: int,
    stockfish_path: str,
) -> tuple[str, dict, list[chess.pgn.Game]]:
    """Evaluate a single model against a single difficulty level."""
    lvl_name = lvl["name"]
    evaluator = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    opponent = UCIAgent(stockfish_path, uci_elo=lvl["elo"])

    wins = draws = losses = 0
    acpl_list: list[float] = []
    game_lengths: list[int] = []
    eval_trajectories: list[list[int]] = []
    pgns: list[chess.pgn.Game] = []

    try:
        for g in range(games_per_level):
            model_color = "white" if g % 2 == 0 else "black"

            gd = (
                play_game(agent, opponent, evaluator)
                if model_color == "white"
                else play_game(opponent, agent, evaluator)
            )
            gd["model_color"] = model_color

            outcome = model_outcome(gd["result"], model_color == "white")
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                draws += 1

            acpl_list.append(compute_acpl(gd, model_color))
            game_lengths.append(gd["total_moves"])
            eval_trajectories.append(gd["evaluations"])

            tag = outcome[0].upper()
            with _log_lock:
                LOGGER.info(
                    "[%s] vs %s  Game %d/%d: %s (%s, %d moves, %s)",
                    model_name,
                    lvl_name,
                    g + 1,
                    games_per_level,
                    gd["result"],
                    tag,
                    gd["total_moves"],
                    gd["termination"],
                )

            pgns.append(create_pgn(gd, f"{model_name} vs SF {lvl_name}"))
    finally:
        opponent.close()
        evaluator.quit()

    total = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / total * 100 if total else 0
    avg_acpl = sum(acpl_list) / len(acpl_list) if acpl_list else 0

    level_result = {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": score_pct,
        "avg_acpl": avg_acpl,
        "acpl_list": acpl_list,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "game_lengths": game_lengths,
        "eval_trajectories": eval_trajectories,
        "elo": lvl["elo"],
    }

    with _log_lock:
        LOGGER.info(
            "[%s] vs %s  => %dW / %dD / %dL  |  Score: %.0f%%  |  Avg ACPL: %.0f",
            model_name,
            lvl_name,
            wins,
            draws,
            losses,
            score_pct,
            avg_acpl,
        )

    return lvl_name, level_result, pgns


def run_evaluation(
    backbone: str,
    checkpoint_path: Path,
    stockfish_path: str,
    games_per_level: int,
    output_dir: Path,
    levels: list[dict] | None = None,
    workers: int = 4,
):
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    pgn_dir = output_dir / "pgn"
    pgn_dir.mkdir(exist_ok=True)

    levels = levels or DIFFICULTY_LEVELS
    models = [backbone]

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
        ", ".join(models),
        len(levels),
        games_per_level,
        len(models) * len(levels) * games_per_level,
        workers,
    )

    results: dict[str, dict[str, dict]] = {}
    all_pgns: list[chess.pgn.Game] = []

    for model_name in models:
        LOGGER.info(
            "=" * 70 + "\n  MODEL: %s  (%d levels in parallel)\n" + "=" * 70,
            model_name.upper(),
            workers,
        )

        try:
            agent = load_model_agent(model_name, checkpoint_path, device)
        except FileNotFoundError as e:
            LOGGER.warning("SKIP — %s", e)
            continue

        results[model_name] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _evaluate_level,
                    agent,
                    model_name,
                    lvl,
                    games_per_level,
                    stockfish_path,
                ): lvl
                for lvl in levels
            }

            for future in as_completed(futures):
                lvl_name, level_result, pgns = future.result()
                results[model_name][lvl_name] = level_result
                all_pgns.extend(pgns)

        # Re-order level keys to match original ordering
        results[model_name] = {
            str(lvl["name"]): results[model_name][lvl["name"]]
            for lvl in levels
            if lvl["name"] in results[model_name]
        }

    # Save PGNs
    pgn_path = pgn_dir / "all_evaluation_games.pgn"
    with open(pgn_path, "w") as f:
        for pgn in all_pgns:
            f.write(str(pgn) + "\n\n")
    LOGGER.info("PGNs saved: %s", pgn_path)

    # Save raw JSON (exclude large trajectory data)
    json_results = {
        m: {
            lvl_name: {k: v for k, v in data.items() if k != "eval_trajectories"}
            for lvl_name, data in lvl_data.items()
        }
        for m, lvl_data in results.items()
    }
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Generate all figures
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_win_rate(results, figures_dir)
    plot_score_pct(results, figures_dir)
    plot_acpl(results, figures_dir)
    plot_acpl_boxplot(results, figures_dir)
    plot_game_length(results, figures_dir)
    plot_result_stacked(results, figures_dir)
    plot_heatmap(results, figures_dir)
    plot_eval_trajectories_from_results(results, figures_dir)
    plot_elo_curve(results, figures_dir)
    generate_report(results, output_dir)

    LOGGER.info("All outputs saved to: %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# 2. Split PGN: split all_evaluation_games.pgn into per-model per-difficulty files
# ---------------------------------------------------------------------------


def split_pgn(input_path: Path, output_dir: Path) -> None:
    """Parse a combined PGN and write per-model-elo files."""
    games: dict[str, list[str]] = {}
    current_lines: list[str] = []
    current_key: str | None = None

    with open(input_path) as f:
        for line in f:
            event_match = re.match(r'^\[Event "(.+)"\]$', line.strip())
            if event_match:
                if current_key and current_lines:
                    games.setdefault(current_key, []).append("".join(current_lines))
                current_lines = [line]
                event = event_match.group(1)
                model, rest = event.split(" vs SF ")
                elo = rest.split("-")[-1]
                current_key = f"{model}_{elo}"
            else:
                current_lines.append(line)

    if current_key and current_lines:
        games.setdefault(current_key, []).append("".join(current_lines))

    output_dir.mkdir(parents=True, exist_ok=True)
    for key, game_list in sorted(games.items()):
        out_path = output_dir / f"{key}.pgn"
        with open(out_path, "w") as f:
            f.write("\n".join(game_list))
        LOGGER.info("%s: %d games", out_path.name, len(game_list))


# ---------------------------------------------------------------------------
# 3. Plot individual games: per-game eval charts compiled into PDFs
# ---------------------------------------------------------------------------


def _parse_individual_game_info(game_text: str) -> dict:
    """Parse a single game for individual game plotting."""
    result = header_value(game_text, "Result") or "unknown"
    white_name = header_value(game_text, "White") or "?"
    black_name = header_value(game_text, "Black") or "?"
    termination = header_value(game_text, "Termination") or "unknown"
    model_is_white = "Learning_" in white_name

    evals = extract_evals(get_move_text(game_text))
    model_side = "White" if model_is_white else "Black"

    if result == "1-0":
        outcome = "Model wins" if model_is_white else "Model loses"
    elif result == "0-1":
        outcome = "Model loses" if model_is_white else "Model wins"
    else:
        outcome = "Draw"

    return {
        "evals": evals,
        "result": result,
        "outcome": outcome,
        "model_side": model_side,
        "white": white_name,
        "black": black_name,
        "termination": termination,
        "model_is_white": model_is_white,
    }


def _plot_single_game(info: dict, game_num: int, out_path: Path) -> None:
    evals = info["evals"]
    if not evals:
        return

    half_moves = np.arange(1, len(evals) + 1) / 2.0
    evals_arr = np.array(evals)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(
        half_moves,
        0,
        evals_arr,
        where=evals_arr >= 0,  # type: ignore
        color="#2ecc71",
        alpha=0.2,
        interpolate=True,
    )
    ax.fill_between(
        half_moves,
        0,
        evals_arr,
        where=evals_arr < 0,  # type: ignore
        color="#e74c3c",
        alpha=0.2,
        interpolate=True,
    )
    ax.plot(half_moves, evals_arr, color="#2c3e50", linewidth=1.5)

    ax.axhline(0, color="#333333", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_xlim(0, half_moves[-1])
    ax.set_ylim(-EVAL_CAP - 0.5, EVAL_CAP + 0.5)
    ax.set_xlabel("Move Number", fontsize=11)
    ax.set_ylabel("Eval (pawns)", fontsize=11)

    outcome_color = (
        "#2ecc71"
        if "wins" in info["outcome"]
        else ("#e74c3c" if "loses" in info["outcome"] else "#95a5a6")
    )
    ax.set_title(
        f"Game {game_num}: {info['outcome']} ({info['result']}, {info['termination']})"
        f" — Model as {info['model_side']}",
        fontsize=12,
        fontweight="bold",
        color=outcome_color,
    )
    ax.grid(True, alpha=0.2, color="#cccccc")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_pdf(
    pdf_path: Path,
    title: str,
    summary: str,
    game_infos: list[dict],
    img_paths: list[Path],
) -> None:
    page_w, page_h = A4
    margin = 20 * mm
    usable_w = page_w - 2 * margin

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h - margin - 10, title)
    c.setFont("Helvetica", 12)
    c.drawString(margin, page_h - margin - 30, summary)

    y = page_h - margin - 55

    for i, (info, img_path) in enumerate(zip(game_infos, img_paths, strict=False), 1):
        text_height, img_height = 100, usable_w * 0.4
        needed = text_height + img_height + 15

        if y - needed < margin:
            c.showPage()
            y = page_h - margin

        c.setFont("Helvetica-Bold", 13)
        outcome = info["outcome"]
        if "wins" in outcome:
            c.setFillColorRGB(0.18, 0.80, 0.44)
        elif "loses" in outcome:
            c.setFillColorRGB(0.91, 0.30, 0.24)
        else:
            c.setFillColorRGB(0.58, 0.65, 0.65)

        c.drawString(margin, y, f"Game {i}: {outcome}")
        c.setFillColorRGB(0, 0, 0)

        y -= 18
        c.setFont("Helvetica", 10)
        for line in [
            f"Result: {info['result']}    Termination: {info['termination']}    "
            f"Moves: {len(info['evals']) // 2}",
            f"Model plays: {info['model_side']}    "
            f"White: {info['white']}    Black: {info['black']}",
        ]:
            c.drawString(margin, y, line)
            y -= 14

        y -= 5
        if img_path.exists():
            c.drawImage(
                str(img_path),
                margin,
                y - img_height,
                width=usable_w,
                height=img_height,
                preserveAspectRatio=True,
                anchor="nw",
            )
            y -= img_height + 15

    c.save()


def plot_individual_games(input_dir: Path, output_dir: Path) -> None:
    """Plot per-game eval curves and compile into PDFs for each split PGN."""
    pgn_files = sorted(input_dir.glob("*.pgn"))
    if not pgn_files:
        LOGGER.warning("No PGN files found in %s", input_dir)
        return

    LOGGER.info("Processing %d PGN files for individual game plots...", len(pgn_files))

    for pgn_path in pgn_files:
        games = parse_games_from_file(pgn_path)
        if not games:
            continue

        stem = pgn_path.stem
        model_name, elo = parse_model_elo_from_stem(stem)
        img_dir = output_dir / stem
        img_dir.mkdir(parents=True, exist_ok=True)

        game_infos: list[dict] = []
        img_paths: list[Path] = []
        wins = losses = draws = 0

        for i, game_text in enumerate(games, 1):
            info = _parse_individual_game_info(game_text)
            if "wins" in info["outcome"]:
                wins += 1
            elif "loses" in info["outcome"]:
                losses += 1
            else:
                draws += 1

            img_path = img_dir / f"game_{i:02d}.png"
            _plot_single_game(info, i, img_path)
            game_infos.append(info)
            img_paths.append(img_path)

        title = f"{model_name} vs Stockfish (Elo {elo})"
        summary = f"{len(games)} games — {wins}W / {losses}L / {draws}D"

        pdf_path = output_dir / f"{stem}.pdf"
        _build_pdf(pdf_path, title, summary, game_infos, img_paths)
        shutil.rmtree(img_dir)

        # Clean up leftover markdown if present
        md_path = output_dir / f"{stem}.md"
        if md_path.exists():
            md_path.unlink()

        LOGGER.info("  %s: %d games -> %s", stem, len(games), pdf_path.name)

    LOGGER.info("Individual game plots done.")


# ---------------------------------------------------------------------------
# 4. Plot eval trajectories: overlaid per-game trajectories with averages
# ---------------------------------------------------------------------------


def plot_trajectory_for_pgn(pgn_path: Path, output_dir: Path) -> None:
    """Plot overlaid eval trajectories for all games in a single PGN file."""
    games = parse_games_from_file(pgn_path)
    if not games:
        return

    stem = pgn_path.stem
    model_name, elo = parse_model_elo_from_stem(stem)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    seen_labels: set[str] = set()
    max_move = 0.0
    traces_by_outcome: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "win": [],
        "loss": [],
        "draw": [],
    }

    for game_text in games:
        move_text = get_move_text(game_text)
        all_evals = extract_evals(move_text)
        if not all_evals:
            continue

        result = header_value(game_text, "Result") or "unknown"
        white = header_value(game_text, "White") or ""
        is_model_white = "Learning_" in white

        half_moves = np.arange(1, len(all_evals) + 1) / 2.0
        max_move = max(max_move, half_moves[-1])
        evals_arr = smooth(np.array(all_evals))

        outcome = model_outcome(result, is_model_white)
        traces_by_outcome[outcome].append((half_moves, evals_arr))

    for outcome in ("win", "loss", "draw"):
        traces = traces_by_outcome[outcome]
        if not traces:
            continue

        color = OUTCOME_COLORS[outcome]
        label = OUTCOME_LABELS[outcome]

        for half_moves, evals_arr in traces:
            lbl = label if label not in seen_labels else None
            if lbl:
                seen_labels.add(label)
            ax.plot(
                half_moves, evals_arr, color=color, alpha=0.15, linewidth=0.8, label=lbl
            )

        # Compute and plot average trajectory
        max_len = max(len(t[1]) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, (_, evals_arr) in enumerate(traces):
            padded[i, : len(evals_arr)] = evals_arr

        mean_evals = np.nanmean(padded, axis=0)
        mean_moves = np.arange(1, max_len + 1) / 2.0
        valid = ~np.isnan(mean_evals)
        ax.plot(
            mean_moves[valid],
            smooth(mean_evals[valid], window=7),
            color=color,
            alpha=0.9,
            linewidth=2.5,
            label=f"{label} (avg)",
        )

    ax.axhline(0, color="#333333", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_xlim(0, max_move)
    ax.set_ylim(-EVAL_CAP - 0.5, EVAL_CAP + 0.5)
    ax.set_xlabel("Move Number", fontsize=12)
    ax.set_ylabel("Evaluation (pawns, white perspective)", fontsize=12)
    ax.set_title(
        f"{model_name} vs Stockfish (Elo {elo}) — Eval Trajectories",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, color="#cccccc")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", framealpha=0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.info("  %s", out_path.name)


def plot_trajectories(input_dir: Path, output_dir: Path) -> None:
    """Plot eval trajectories for every split PGN file."""
    pgn_files = sorted(input_dir.glob("*.pgn"))
    if not pgn_files:
        LOGGER.warning("No PGN files found in %s", input_dir)
        return

    LOGGER.info("Plotting %d PGN trajectory files...", len(pgn_files))
    for pgn_file in pgn_files:
        plot_trajectory_for_pgn(pgn_file, output_dir)
    LOGGER.info("Trajectory plots done.")


# ---------------------------------------------------------------------------
# Evaluation plotting helpers (from evaluate_all)
# ---------------------------------------------------------------------------


def _level_names(results: dict) -> list[str]:
    for model_data in results.values():
        return list(model_data.keys())
    return []


def _elo_ticks(results: dict) -> list[int]:
    for model_data in results.values():
        return [d["elo"] for d in model_data.values()]
    return []


def _styled_figure(nrows=1, ncols=1, figsize=None):
    figsize = figsize or ((12, 6) if ncols == 1 else (14, 6 * nrows))
    return plt.subplots(nrows, ncols, figsize=figsize)


def plot_win_rate(results: dict, out: Path):
    fig, ax = _styled_figure()
    levels, elos = _level_names(results), _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        rates = [
            level_data[lvl]["wins"]
            / max(
                level_data[lvl]["wins"]
                + level_data[lvl]["draws"]
                + level_data[lvl]["losses"],
                1,
            )
            * 100
            for lvl in levels
        ]
        ax.plot(
            x,
            rates,
            "o-",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=2,
            markersize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in elos], fontsize=9)
    ax.set_xlabel("Opponent Elo", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate vs Opponent Strength", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(out / "win_rate.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_pct(results: dict, out: Path):
    fig, ax = _styled_figure()
    levels, elos = _level_names(results), _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        scores = [level_data[lvl]["score_pct"] for lvl in levels]
        ax.plot(
            x,
            scores,
            "s-",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=2,
            markersize=6,
        )

    ax.axhline(
        y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="50% baseline"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in elos], fontsize=9)
    ax.set_xlabel("Opponent Elo", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(
        "Score Percentage vs Opponent Strength (W=1, D=0.5, L=0)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(out / "score_percentage.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_acpl(results: dict, out: Path):
    fig, ax = _styled_figure()
    levels, elos = _level_names(results), _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        acpls = [level_data[lvl]["avg_acpl"] for lvl in levels]
        ax.plot(
            x,
            acpls,
            "^-",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=2,
            markersize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in elos], fontsize=9)
    ax.set_xlabel("Opponent Elo", fontsize=12)
    ax.set_ylabel("Average Centipawn Loss", fontsize=12)
    ax.set_title(
        "Average Centipawn Loss vs Opponent Strength", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "avg_centipawn_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_acpl_boxplot(results: dict, out: Path):
    models_present = [m for m in MODELS if m in results]
    levels = _level_names(results)
    n_models = len(models_present)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, model in enumerate(models_present):
        ax = axes[i]
        data_to_plot = [results[model][lvl]["acpl_list"] for lvl in levels]
        tick_labels = [str(results[model][lvl]["elo"]) for lvl in levels]

        bp = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.6)

        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("Opponent Elo", fontsize=9)
        if i == 0:
            ax.set_ylabel("Centipawn Loss", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "ACPL Distribution per Difficulty Level", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out / "acpl_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_game_length(results: dict, out: Path):
    fig, ax = _styled_figure()
    levels, elos = _level_names(results), _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        lengths = [level_data[lvl]["avg_game_length"] for lvl in levels]
        ax.plot(
            x,
            lengths,
            "D-",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=2,
            markersize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in elos], fontsize=9)
    ax.set_xlabel("Opponent Elo", fontsize=12)
    ax.set_ylabel("Average Game Length (moves)", fontsize=12)
    ax.set_title(
        "Average Game Length vs Opponent Strength", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "game_length.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_result_stacked(results: dict, out: Path):
    models_present = [m for m in MODELS if m in results]
    levels, elos = _level_names(results), _elo_ticks(results)
    n_levels = len(levels)

    fig, axes = plt.subplots(
        1, len(models_present), figsize=(4 * len(models_present), 6), sharey=True
    )
    if len(models_present) == 1:
        axes = [axes]

    for idx, model in enumerate(models_present):
        ax = axes[idx]
        w = [results[model][lvl]["wins"] for lvl in levels]
        d = [results[model][lvl]["draws"] for lvl in levels]
        lo = [results[model][lvl]["losses"] for lvl in levels]
        total = [w[i] + d[i] + lo[i] for i in range(n_levels)]
        wp = [w[i] / max(t, 1) * 100 for i, t in enumerate(total)]
        dp = [d[i] / max(t, 1) * 100 for i, t in enumerate(total)]
        lp = [lo[i] / max(t, 1) * 100 for i, t in enumerate(total)]

        x = np.arange(n_levels)
        ax.bar(x, wp, color="#2ecc71", label="Win")
        ax.bar(x, dp, bottom=wp, color="#95a5a6", label="Draw")
        ax.bar(
            x,
            lp,
            bottom=[wp[i] + dp[i] for i in range(n_levels)],
            color="#e74c3c",
            label="Loss",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in elos], fontsize=7, rotation=45)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("Opponent Elo", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Percentage", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle("Win / Draw / Loss Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "result_stacked.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(results: dict, out: Path):
    models_present = [m for m in MODELS if m in results]
    levels, elos = _level_names(results), _elo_ticks(results)

    score_matrix = np.zeros((len(models_present), len(levels)))
    for i, model in enumerate(models_present):
        for j, lvl in enumerate(levels):
            score_matrix[i, j] = results[model][lvl]["score_pct"]

    fig, ax = plt.subplots(
        figsize=(max(10, len(levels) * 1.2), max(5, len(models_present) * 0.9))
    )
    im = ax.imshow(score_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([str(e) for e in elos], fontsize=9)
    ax.set_yticks(np.arange(len(models_present)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models_present], fontsize=10)

    for i in range(len(models_present)):
        for j in range(len(levels)):
            val = score_matrix[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(
                j,
                i,
                f"{val:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Opponent Elo", fontsize=12)
    ax.set_title("Score % Heatmap (Model × Difficulty)", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score %", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "score_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_eval_trajectories_from_results(results: dict, out: Path):
    models_present = [m for m in MODELS if m in results]
    levels = _level_names(results)

    sample_indices = [0, len(levels) // 2, len(levels) - 1]
    sample_levels = [levels[i] for i in sample_indices if i < len(levels)]

    fig, axes = plt.subplots(1, len(sample_levels), figsize=(7 * len(sample_levels), 5))
    if len(sample_levels) == 1:
        axes = [axes]

    for ax_idx, lvl in enumerate(sample_levels):
        ax = axes[ax_idx]
        for model in models_present:
            trajs = results[model][lvl]["eval_trajectories"]
            if not trajs:
                continue

            max_len = max(len(t) for t in trajs)
            avg_traj = [
                np.mean([t[mi] / 100 for t in trajs if mi < len(t)])
                for mi in range(max_len)
            ]
            avg_traj_clipped = np.clip(avg_traj, -10, 10)
            ax.plot(
                avg_traj_clipped,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=1.5,
                alpha=0.8,
            )

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("Move Number", fontsize=10)
        ax.set_ylabel("Avg Eval (pawns)", fontsize=10)
        elo_val = results[models_present[0]][lvl]["elo"]
        ax.set_title(f"vs ~{elo_val} Elo", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Average Evaluation Trajectory (Model as White/Black)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out / "eval_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_elo_curve(results: dict, out: Path):
    models_present = [m for m in MODELS if m in results]
    levels, elos = _level_names(results), _elo_ticks(results)
    x_data = np.array(elos)
    
    def logistic_function(x, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models_present:
        y_data = np.array([results[model][lvl]["score_pct"] / 100.0 for lvl in levels])
        
        ax.scatter(x_data, y_data, color=MODEL_COLORS[model], alpha=0.5)
        
        if np.max(y_data) == 0.0 or np.min(y_data) == 1.0:
            continue
            
        try:
            popt, _ = optimize.curve_fit(logistic_function, x_data, y_data, p0=[-0.01, 1500])
            estimated_elo = popt[1]
            x_smooth = np.linspace(min(x_data), max(x_data), 200)
            y_smooth = logistic_function(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, color=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]} (Elo: {estimated_elo:.0f})")
        except RuntimeError:
            pass

    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.set_xlabel("Opponent Elo", fontsize=12)
    ax.set_ylabel("Score %", fontsize=12)
    ax.set_title("Estimated Playing Elo via Logistic Fit", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "estimated_elo_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(results: dict, output_dir: Path):
    models_present = [m for m in MODELS if m in results]
    levels = _level_names(results)

    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(f"""\
# Comprehensive Model Evaluation Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Checkpoint source:** runs/50_10M
**Evaluation depth:** {EVAL_DEPTH}
**Models tested:** {len(models_present)}
**Difficulty levels:** {len(levels)}

## Difficulty Levels

| Level | ~Elo |
|-------|------|
""")
        for lvl in DIFFICULTY_LEVELS:
            f.write(
                f"| {lvl['name']} | {lvl['elo']} |\n"
            )

        f.write("\n## Estimated Playing Elo\n\n")
        f.write("| Model | Estimated Playing Elo | Logistic Fit k |\n")
        f.write("|-------|----------------------|----------------|\n")
        for model in models_present:
            scores = np.array([results[model][lvl]['score_pct'] / 100.0 for lvl in levels])
            x_data = np.array([results[model][lvl]['elo'] for lvl in levels])
            
            def logistic_function(x, k, x0):
                return 1.0 / (1.0 + np.exp(-k * (x - x0)))
                
            if np.max(scores) == 0.0:
                est_elo = 0
                k_val = 0
            elif np.min(scores) == 1.0:
                est_elo = 4000
                k_val = 0
            else:
                try:
                    popt, _ = optimize.curve_fit(logistic_function, x_data, scores, p0=[-0.01, 1500])
                    est_elo = popt[1]
                    k_val = popt[0]
                except RuntimeError:
                    est_elo = 1500
                    k_val = 0
            
            f.write(f"| {MODEL_LABELS[model]} | {est_elo:.1f} | {k_val:.4f} |\n")

        f.write("\n## Score Percentage Summary\n\n")
        header = (
            "| Model | "
            + " | ".join(str(results[models_present[0]][lvl]["elo"]) for lvl in levels)
            + " |\n"
        )
        sep = "|-------|" + "|".join("-------" for _ in levels) + "|\n"
        f.write(header)
        f.write(sep)
        for model in models_present:
            row = f"| {MODEL_LABELS[model]} | "
            row += " | ".join(f"{results[model][lvl]['score_pct']:.0f}%" for lvl in levels)
            f.write(row + " |\n")

        f.write("\n## Average Centipawn Loss\n\n")
        f.write(header)
        f.write(sep)
        for model in models_present:
            row = f"| {MODEL_LABELS[model]} | "
            row += " | ".join(f"{results[model][lvl]['avg_acpl']:.0f}" for lvl in levels)
            f.write(row + " |\n")

        f.write("\n## Win/Draw/Loss Detail\n\n")
        for model in models_present:
            f.write(f"### {MODEL_LABELS[model]}\n\n")
            f.write("| Opponent Elo | W | D | L | Score % | Avg ACPL | Avg Length |\n")
            f.write("|-------------|---|---|---|---------|----------|------------|\n")
            for lvl in levels:
                lvl_data = dict(results[model][lvl])
                f.write(
                    f"| {lvl_data['elo']} | {lvl_data['wins']} | {lvl_data['draws']} "
                    f"| {lvl_data['losses']} | {lvl_data['score_pct']:.0f}% "
                    f"| {lvl_data['avg_acpl']:.0f} | {lvl_data['avg_game_length']:.0f} |\n"
                )
            f.write("\n")

        f.write("## Figures\n\n")
        for fig_name, caption in [
            ("win_rate.png", "Win Rate vs Opponent Strength"),
            ("score_percentage.png", "Score % vs Opponent Strength"),
            ("estimated_elo_curve.png", "Estimated Playing Elo Curve"),
            ("avg_centipawn_loss.png", "Average Centipawn Loss"),
            ("acpl_boxplot.png", "ACPL Distribution Boxplots"),
            ("game_length.png", "Average Game Length"),
            ("result_stacked.png", "Win/Draw/Loss Stacked Bars"),
            ("score_heatmap.png", "Score % Heatmap"),
            ("eval_trajectories.png", "Evaluation Trajectories"),
        ]:
            f.write(f"### {caption}\n\n![{caption}](figures/{fig_name})\n\n")

    LOGGER.info("Report saved: %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run comprehensive Elo benchmark pipeline for a chess model.",
    )
    parser.add_argument("--backbone", type=str, default="resnet", help="Network backbone architecture")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--games", type=int, default=4, help="Games per difficulty level (use even number)")
    parser.add_argument("--output-dir", type=str, default="runs/evaluation")
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated level indices (0-9)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel threads (each spawns its own Stockfish)")

    args = parser.parse_args()

    levels = DIFFICULTY_LEVELS
    if args.levels:
        indices = [int(x.strip()) for x in args.levels.split(",")]
        levels = [DIFFICULTY_LEVELS[i] for i in indices]

    output_dir = Path(args.output_dir)
    pgn_dir = output_dir / "pgn"
    split_dir = pgn_dir / "split"
    figures_dir = output_dir / "figures"

    LOGGER.info("=" * 70 + "\n  COMPREHENSIVE MODEL EVALUATION\n" + "=" * 70)

    # Step 1: evaluate
    LOGGER.info("Step 1/4: Running evaluation...")
    run_evaluation(
        backbone=args.backbone,
        checkpoint_path=Path(args.weights),
        stockfish_path=args.stockfish,
        games_per_level=args.games,
        output_dir=output_dir,
        levels=levels,
        workers=args.workers,
    )

    # Step 2: split
    pgn_path = pgn_dir / "all_evaluation_games.pgn"
    if pgn_path.exists():
        LOGGER.info("Step 2/4: Splitting PGN...")
        split_pgn(pgn_path, split_dir)
    else:
        LOGGER.warning("Skipping split — %s not found", pgn_path)

    # Step 3: individual game plots
    if split_dir.exists():
        LOGGER.info("Step 3/4: Plotting individual games...")
        plot_individual_games(split_dir, figures_dir / "individual_games")
    else:
        LOGGER.warning("Skipping individual game plots — %s not found", split_dir)

    # Step 4: trajectory plots
    if split_dir.exists():
        LOGGER.info("Step 4/4: Plotting eval trajectories...")
        plot_trajectories(split_dir, figures_dir / "trajectories")
    else:
        LOGGER.warning("Skipping trajectory plots — %s not found", split_dir)

    LOGGER.info("=" * 70 + "\n  EVALUATION COMPLETE!\n" + "=" * 70)
    LOGGER.info("Full pipeline complete. Outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
