"""Plot per-game evaluation trajectories for each split PGN file.

Usage:
    python scripts/plot_eval_trajectories.py \
        --input-dir runs/50_10M/evaluation/pgn/split \
        --output-dir runs/50_10M/evaluation/figures/trajectories
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_PATTERN = re.compile(r"\[%eval ([^\]]+)\]")
EVAL_CAP = 10.0
SMOOTH_WINDOW = 5


def parse_game_info(game_text: str) -> tuple[list[float], str, bool]:
    """Extract half-move evals, result, and whether the model plays white.

    Returns (all_evals, result, model_is_white).
    All evals are from white's perspective, clamped to ±EVAL_CAP.
    Checkmate (#0) is set to ±EVAL_CAP matching the winning side.
    """
    result = "unknown"
    result_match = re.search(r'\[Result "([^"]+)"\]', game_text)
    if result_match:
        result = result_match.group(1)

    white_match = re.search(r'\[White "([^"]+)"\]', game_text)
    model_is_white = bool(white_match and "Learning_" in white_match.group(1))

    lines = game_text.split("\n")
    move_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            move_line += " " + stripped

    raw_evals = EVAL_PATTERN.findall(move_line)

    all_evals: list[float] = []
    for i, raw in enumerate(raw_evals):
        is_white_move = i % 2 == 0

        if raw.startswith("#"):
            val = EVAL_CAP if is_white_move else -EVAL_CAP
        elif raw.startswith("-#"):
            val = -EVAL_CAP
        else:
            val = float(raw)
            val = max(-EVAL_CAP, min(EVAL_CAP, val))

        all_evals.append(val)

    return all_evals, result, model_is_white


def parse_games(pgn_path: Path) -> list[str]:
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


def model_outcome(result: str, model_is_white: bool) -> str:
    if result == "1-0":
        return "win" if model_is_white else "loss"
    elif result == "0-1":
        return "loss" if model_is_white else "win"
    return "draw"


OUTCOME_COLORS = {"win": "#2ecc71", "loss": "#e74c3c", "draw": "#95a5a6"}
OUTCOME_LABELS = {"win": "Model wins", "loss": "Model loses", "draw": "Draw"}


def smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def plot_pgn(pgn_path: Path, output_dir: Path) -> None:
    games = parse_games(pgn_path)
    if not games:
        return

    stem = pgn_path.stem
    parts = stem.rsplit("_", 1)
    model_name = parts[0].replace("_", " ").title()
    elo = parts[1] if len(parts) == 2 else "?"

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
        all_evals, result, model_is_white = parse_game_info(game_text)
        if not all_evals:
            continue

        half_moves = np.arange(1, len(all_evals) + 1) / 2.0
        max_move = max(max_move, half_moves[-1])
        evals_arr = smooth(np.array(all_evals))

        outcome = model_outcome(result, model_is_white)
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

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", framealpha=0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot eval trajectories for split PGN files"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory with split PGN files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for figures"
    )
    args = parser.parse_args()

    pgn_files = sorted(args.input_dir.glob("*.pgn"))
    if not pgn_files:
        print(f"No PGN files found in {args.input_dir}")
        return

    print(f"Plotting {len(pgn_files)} PGN files...")
    for pgn_file in pgn_files:
        plot_pgn(pgn_file, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
