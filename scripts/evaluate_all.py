#!/usr/bin/env python3
"""
Comprehensive model evaluation across escalating Stockfish difficulty levels.

Tests all 6 models from runs/50_10M against Stockfish at progressively harder
settings, collecting win rates, centipawn loss, game length, and more.
Produces publication-quality graphs.

Difficulty levels based on community-tested Elo mappings:
  Level  | Skill | Depth | ~Elo (CCRL) | Label
  -------|-------|-------|-------------|------------------
  1      |   0   |   1   |  ~800       | Absolute beginner
  2      |   1   |   2   |  ~1100      | Beginner
  3      |   3   |   3   |  ~1400      | Casual
  4      |   5   |   5   |  ~1700      | Club player
  5      |   7   |   5   |  ~2000      | Strong club
  6      |   9   |   8   |  ~2300      | Expert / candidate master
  7      |  11   |   8   |  ~2500      | Master-level
  8      |  14   |  10   |  ~2800      | Strong master
  9      |  17   |  13   |  ~3100      | Super GM engine
  10     |  20   |  18   |  ~3200+     | Full strength
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.learning_agent import LearningAgent
from src.agents.uci_agent import UCIAgent
from src.config import settings
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model

_print_lock = threading.Lock()


MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]

DIFFICULTY_LEVELS = [
    {"name": "Beginner-800", "skill": 0, "depth": 1, "elo": 800},
    {"name": "Novice-1100", "skill": 1, "depth": 2, "elo": 1100},
    {"name": "Casual-1400", "skill": 3, "depth": 3, "elo": 1400},
    {"name": "Club-1700", "skill": 5, "depth": 5, "elo": 1700},
    {"name": "Strong-2000", "skill": 7, "depth": 5, "elo": 2000},
    {"name": "Expert-2300", "skill": 9, "depth": 8, "elo": 2300},
    {"name": "Master-2500", "skill": 11, "depth": 8, "elo": 2500},
    {"name": "IM-2800", "skill": 14, "depth": 10, "elo": 2800},
    {"name": "GM-3100", "skill": 17, "depth": 13, "elo": 3100},
    {"name": "Full-3200", "skill": 20, "depth": 18, "elo": 3200},
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


def load_model_agent(
    model_name: str, checkpoint_dir: Path, device: torch.device
) -> LearningAgent:
    model_cfg = settings.model.model_copy(
        update={"backbone": model_name, "head": "dual"}
    )

    model = create_model(model_cfg)
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory()

    return LearningAgent(model=model, encoder=encoder, device=device, temperature=0.0)


def evaluate_position(
    engine: chess.engine.SimpleEngine, board: chess.Board, depth: int = EVAL_DEPTH
) -> int:
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            mate_in = score.mate()
            return (10000 - mate_in * 10) if mate_in > 0 else (-10000 - mate_in * 10)
        return score.score()
    except Exception:
        return 0


def play_game(
    white_agent,
    black_agent,
    evaluator: chess.engine.SimpleEngine,
    max_moves: int = 200,
) -> dict:
    board = chess.Board()
    data = {
        "moves": [],
        "san_moves": [],
        "evaluations": [],
        "white_agent": getattr(white_agent, "name", str(white_agent)),
        "black_agent": getattr(black_agent, "name", str(black_agent)),
    }

    data["evaluations"].append(evaluate_position(evaluator, board))

    while not board.is_game_over() and len(data["moves"]) < max_moves:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        move = agent.get_move(board)
        data["san_moves"].append(board.san(move))
        data["moves"].append(move.uci())
        board.push(move)
        data["evaluations"].append(evaluate_position(evaluator, board))

    if board.is_checkmate():
        data["result"] = "0-1" if board.turn == chess.WHITE else "1-0"
        data["termination"] = "checkmate"
    elif board.is_stalemate():
        data["result"] = "1/2-1/2"
        data["termination"] = "stalemate"
    elif board.is_insufficient_material():
        data["result"] = "1/2-1/2"
        data["termination"] = "insufficient_material"
    elif board.is_fifty_moves():
        data["result"] = "1/2-1/2"
        data["termination"] = "fifty_moves"
    elif board.is_repetition():
        data["result"] = "1/2-1/2"
        data["termination"] = "repetition"
    elif len(data["moves"]) >= max_moves:
        data["result"] = "1/2-1/2"
        data["termination"] = "max_moves"
    else:
        data["result"] = "*"
        data["termination"] = "unknown"

    data["total_moves"] = len(data["moves"])
    data["final_fen"] = board.fen()
    return data


def compute_acpl(game_data: dict, model_color: str) -> float:
    """Average centipawn loss for the model's moves."""
    evals = game_data["evaluations"]
    losses = []
    for i in range(len(evals) - 1):
        move_num = i  # 0-indexed: move 0 is white's first, move 1 is black's first
        is_white_move = move_num % 2 == 0
        if (model_color == "white" and is_white_move) or (
            model_color == "black" and not is_white_move
        ):
            before = evals[i]
            after = evals[i + 1]
            loss = before - after if model_color == "white" else after - before
            losses.append(max(loss, 0))
    return sum(losses) / len(losses) if losses else 0.0


def create_pgn(game_data: dict, event: str) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = game_data["white_agent"]
    game.headers["Black"] = game_data["black_agent"]
    game.headers["Result"] = game_data["result"]
    game.headers["Termination"] = game_data.get("termination", "unknown")

    node = game
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


# ---------------------------------------------------------------------------
# Per-level worker (runs in its own thread with its own Stockfish instances)
# ---------------------------------------------------------------------------


def _evaluate_level(
    agent: LearningAgent,
    model_name: str,
    lvl: dict,
    games_per_level: int,
    stockfish_path: str,
) -> tuple[str, dict, list[chess.pgn.Game]]:
    """Evaluate a single model against a single difficulty level.

    Each call spawns its own evaluator + opponent Stockfish processes so
    multiple levels can run in parallel without contention.
    """
    lvl_name = lvl["name"]
    evaluator = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    opponent = UCIAgent(stockfish_path, depth=lvl["depth"], skill_level=lvl["skill"])

    wins, draws, losses = 0, 0, 0
    acpl_list: list[float] = []
    game_lengths: list[int] = []
    eval_trajectories: list[list[int]] = []
    pgns: list[chess.pgn.Game] = []

    try:
        for g in range(games_per_level):
            if g % 2 == 0:
                gd = play_game(agent, opponent, evaluator)
                model_color = "white"
            else:
                gd = play_game(opponent, agent, evaluator)
                model_color = "black"

            gd["model_color"] = model_color

            if gd["result"] == "1-0":
                if model_color == "white":
                    wins += 1
                else:
                    losses += 1
            elif gd["result"] == "0-1":
                if model_color == "black":
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1

            acpl_list.append(compute_acpl(gd, model_color))
            game_lengths.append(gd["total_moves"])
            eval_trajectories.append(gd["evaluations"])

            tag = (
                "W"
                if (gd["result"] == "1-0" and model_color == "white")
                or (gd["result"] == "0-1" and model_color == "black")
                else (
                    "L"
                    if (gd["result"] == "1-0" and model_color == "black")
                    or (gd["result"] == "0-1" and model_color == "white")
                    else "D"
                )
            )
            with _print_lock:
                print(
                    f"  [{model_name}] vs {lvl_name}  Game {g + 1}/{games_per_level}: {gd['result']} ({tag}, {gd['total_moves']} moves, {gd['termination']})"
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
        "skill": lvl["skill"],
        "depth": lvl["depth"],
    }

    with _print_lock:
        print(
            f"  [{model_name}] vs {lvl_name}  => {wins}W / {draws}D / {losses}L  |  Score: {score_pct:.0f}%  |  Avg ACPL: {avg_acpl:.0f}"
        )

    return lvl_name, level_result, pgns


# ---------------------------------------------------------------------------
# Main evaluation loop (threaded)
# ---------------------------------------------------------------------------


def run_evaluation(
    checkpoint_dir: Path,
    stockfish_path: str,
    games_per_level: int,
    output_dir: Path,
    levels: list[dict] | None = None,
    models: list[str] | None = None,
    workers: int = 4,
):
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    pgn_dir = output_dir / "pgn"
    pgn_dir.mkdir(exist_ok=True)

    levels = levels or DIFFICULTY_LEVELS
    models = models or MODELS

    print(f"Device: {device}")
    print(f"Models: {', '.join(models)}")
    print(f"Difficulty levels: {len(levels)}")
    print(f"Games per model per level: {games_per_level}")
    print(f"Total games: {len(models) * len(levels) * games_per_level}")
    print(f"Worker threads: {workers}")
    print()

    results: dict[str, dict[str, dict]] = {}
    all_pgns: list[chess.pgn.Game] = []

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"  MODEL: {model_name.upper()}  ({workers} levels in parallel)")
        print(f"{'=' * 70}")

        try:
            agent = load_model_agent(model_name, checkpoint_dir, device)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
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

        # Re-order level keys to match the original level ordering
        ordered = {}
        for lvl in levels:
            if lvl["name"] in results[model_name]:
                ordered[lvl["name"]] = results[model_name][lvl["name"]]
        results[model_name] = ordered

    # Save PGNs
    pgn_path = pgn_dir / "all_evaluation_games.pgn"
    with open(pgn_path, "w") as f:
        for pgn in all_pgns:
            f.write(str(pgn))
            f.write("\n\n")
    print(f"\nPGNs saved: {pgn_path}")

    # Save raw JSON
    json_results = {}
    for m in results:
        json_results[m] = {}
        for lvl_name, data in results[m].items():
            json_results[m][lvl_name] = {
                k: v for k, v in data.items() if k not in ("eval_trajectories",)
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
    plot_eval_trajectories(results, figures_dir)
    generate_report(results, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
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
    if figsize is None:
        figsize = (12, 6) if ncols == 1 else (14, 6 * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, ax


def plot_win_rate(results: dict, out: Path):
    fig, ax = _styled_figure()
    levels = _level_names(results)
    elos = _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        rates = [
            level_data[l]["wins"]
            / max(
                level_data[l]["wins"]
                + level_data[l]["draws"]
                + level_data[l]["losses"],
                1,
            )
            * 100
            for l in levels
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
    ax.set_xticklabels([f"{e}" for e in elos], fontsize=9)
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
    levels = _level_names(results)
    elos = _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        scores = [level_data[l]["score_pct"] for l in levels]
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
    ax.set_xticklabels([f"{e}" for e in elos], fontsize=9)
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
    levels = _level_names(results)
    elos = _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        acpls = [level_data[l]["avg_acpl"] for l in levels]
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
    ax.set_xticklabels([f"{e}" for e in elos], fontsize=9)
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
        data_to_plot = []
        tick_labels = []
        for lvl in levels:
            d = results[model][lvl]
            data_to_plot.append(d["acpl_list"])
            tick_labels.append(str(d["elo"]))
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
    levels = _level_names(results)
    elos = _elo_ticks(results)
    x = np.arange(len(levels))

    for model, level_data in results.items():
        lengths = [level_data[l]["avg_game_length"] for l in levels]
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
    ax.set_xticklabels([f"{e}" for e in elos], fontsize=9)
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
    levels = _level_names(results)
    elos = _elo_ticks(results)

    n_levels = len(levels)
    fig, axes = plt.subplots(
        1, len(models_present), figsize=(4 * len(models_present), 6), sharey=True
    )
    if len(models_present) == 1:
        axes = [axes]

    for idx, model in enumerate(models_present):
        ax = axes[idx]
        w = [results[model][l]["wins"] for l in levels]
        d = [results[model][l]["draws"] for l in levels]
        lo = [results[model][l]["losses"] for l in levels]
        x = np.arange(n_levels)
        total = [w[i] + d[i] + lo[i] for i in range(n_levels)]
        wp = [w[i] / max(t, 1) * 100 for i, t in enumerate(total)]
        dp = [d[i] / max(t, 1) * 100 for i, t in enumerate(total)]
        lp = [lo[i] / max(t, 1) * 100 for i, t in enumerate(total)]

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
    levels = _level_names(results)
    elos = _elo_ticks(results)

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


def plot_eval_trajectories(results: dict, out: Path):
    models_present = [m for m in MODELS if m in results]
    levels = _level_names(results)

    # Pick first, middle, and last difficulty level
    sample_indices = [0, len(levels) // 2, len(levels) - 1]
    sample_levels = [levels[i] for i in sample_indices if i < len(levels)]

    fig, axes = plt.subplots(1, len(sample_levels), figsize=(7 * len(sample_levels), 5))
    if len(sample_levels) == 1:
        axes = [axes]

    for ax_idx, lvl in enumerate(sample_levels):
        ax = axes[ax_idx]
        for model in models_present:
            trajs = results[model][lvl]["eval_trajectories"]
            if trajs:
                avg_traj = []
                max_len = max(len(t) for t in trajs)
                for move_idx in range(max_len):
                    vals = [t[move_idx] / 100 for t in trajs if move_idx < len(t)]
                    avg_traj.append(np.mean(vals))
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


def generate_report(results: dict, output_dir: Path):
    models_present = [m for m in MODELS if m in results]
    levels = _level_names(results)

    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Comprehensive Model Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Checkpoint source:** runs/50_10M\n")
        f.write(f"**Evaluation depth:** {EVAL_DEPTH}\n")
        f.write(f"**Models tested:** {len(models_present)}\n")
        f.write(f"**Difficulty levels:** {len(levels)}\n\n")

        f.write("## Difficulty Levels\n\n")
        f.write("| Level | Skill | Depth | ~Elo |\n")
        f.write("|-------|-------|-------|------|\n")
        for lvl in DIFFICULTY_LEVELS:
            f.write(
                f"| {lvl['name']} | {lvl['skill']} | {lvl['depth']} | {lvl['elo']} |\n"
            )

        f.write("\n## Score Percentage Summary\n\n")
        header = (
            "| Model | "
            + " | ".join(str(results[models_present[0]][l]["elo"]) for l in levels)
            + " |\n"
        )
        sep = "|-------|" + "|".join("-------" for _ in levels) + "|\n"
        f.write(header)
        f.write(sep)
        for model in models_present:
            row = f"| {MODEL_LABELS[model]} | "
            row += " | ".join(f"{results[model][l]['score_pct']:.0f}%" for l in levels)
            row += " |\n"
            f.write(row)

        f.write("\n## Average Centipawn Loss\n\n")
        f.write(header)
        f.write(sep)
        for model in models_present:
            row = f"| {MODEL_LABELS[model]} | "
            row += " | ".join(f"{results[model][l]['avg_acpl']:.0f}" for l in levels)
            row += " |\n"
            f.write(row)

        f.write("\n## Win/Draw/Loss Detail\n\n")
        for model in models_present:
            f.write(f"### {MODEL_LABELS[model]}\n\n")
            f.write("| Opponent Elo | W | D | L | Score % | Avg ACPL | Avg Length |\n")
            f.write("|-------------|---|---|---|---------|----------|------------|\n")
            for l in levels:
                d = results[model][l]
                f.write(
                    f"| {d['elo']} | {d['wins']} | {d['draws']} | {d['losses']} | {d['score_pct']:.0f}% | {d['avg_acpl']:.0f} | {d['avg_game_length']:.0f} |\n"
                )
            f.write("\n")

        f.write("## Figures\n\n")
        for fig_name, caption in [
            ("win_rate.png", "Win Rate vs Opponent Strength"),
            ("score_percentage.png", "Score % vs Opponent Strength"),
            ("avg_centipawn_loss.png", "Average Centipawn Loss"),
            ("acpl_boxplot.png", "ACPL Distribution Boxplots"),
            ("game_length.png", "Average Game Length"),
            ("result_stacked.png", "Win/Draw/Loss Stacked Bars"),
            ("score_heatmap.png", "Score % Heatmap"),
            ("eval_trajectories.png", "Evaluation Trajectories"),
        ]:
            f.write(f"### {caption}\n\n")
            f.write(f"![{caption}](figures/{fig_name})\n\n")

    print(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models across escalating Stockfish difficulty"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="runs/50_10M")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument(
        "--games",
        type=int,
        default=4,
        help="Games per model per difficulty level (use even number)",
    )
    parser.add_argument("--output-dir", type=str, default="runs/50_10M/evaluation")
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated level indices to run (0-9). E.g. '0,1,2,3' for easy levels only.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to test. E.g. 'resnet,gat'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel threads (each spawns its own Stockfish). "
        "M1 Pro: 4-6 recommended.",
    )

    args = parser.parse_args()

    levels = DIFFICULTY_LEVELS
    if args.levels:
        indices = [int(x.strip()) for x in args.levels.split(",")]
        levels = [DIFFICULTY_LEVELS[i] for i in indices]

    models = MODELS
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    print("=" * 70)
    print("  COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    run_evaluation(
        checkpoint_dir=Path(args.checkpoint_dir),
        stockfish_path=args.stockfish,
        games_per_level=args.games,
        output_dir=Path(args.output_dir),
        levels=levels,
        models=models,
        workers=args.workers,
    )

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
