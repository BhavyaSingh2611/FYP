"""
Elo-based Benchmark using Lichess Puzzles via Stratified Sampling and Logistic Curve Fitting.
"""

import argparse
import logging
from pathlib import Path

import chess
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import optimize

from src.agents.learning_agent import LearningAgent
from src.models.factory import create_model, get_encoder_for_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def logistic_function(x, k, x0):
    """Logistic function for curve fitting."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def evaluate_puzzle(agent: LearningAgent, fen: str, moves_str: str) -> int:
    """
    Evaluates if the agent can solve the Lichess puzzle.

    Args:
        agent: LearningAgent to generate moves.
        fen: Starting FEN (before the opponent's first move).
        moves_str: Space-separated UCI move sequence.
                   Example: "e8f7 e2e6 f7f8 e6f7"

    Returns:
        1 if agent successfully played all its expected moves, 0 otherwise.
    """
    board = chess.Board(fen)
    moves = moves_str.strip().split()

    if not moves:
        return 0

    # 1. Opponent plays the first move
    opponent_first_move = chess.Move.from_uci(moves[0])
    board.push(opponent_first_move)

    # 2. Iterate through the rest
    for i in range(1, len(moves)):
        expected_move = chess.Move.from_uci(moves[i])

        # If it's our turn
        if i % 2 != 0:
            agent_move = agent.get_move(board)
            if agent_move != expected_move:
                return 0

        # Push the expected move regardless (to continue sequence if opponent's turn)
        # Or if we're evaluating our turn and we matched exactly.
        board.push(expected_move)

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a chess model's Elo using Lichess Puzzles via Stratified Sampling."
    )
    parser.add_argument("--backbone", type=str, default="resnet", help="Network backbone architecture")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pt). If None, uses randomly initialized weights.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/puzzles/*.parquet",
        help="Path or glob pattern to Lichess puzzle parquet files.",
    )
    parser.add_argument("--min-elo", type=int, default=800, help="Starting Elo bracket.")
    parser.add_argument("--max-elo", type=int, default=2800, help="Ending Elo bracket.")
    parser.add_argument("--step-elo", type=int, default=100, help="Step size between Elo brackets.")
    parser.add_argument(
        "--puzzles-per-bracket", type=int, default=20, help="Number of puzzles to evaluate per bracket."
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load Model
    encoder_cls = get_encoder_for_model(args.backbone)
    encoder = encoder_cls()
    model = create_model(args.backbone)

    if args.weights:
        state_dict = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Handle torch.compile prefix
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                cleaned_state_dict[k[len("_orig_mod.") :]] = v
            else:
                cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict)
        logging.info(f"Loaded weights from {args.weights}")
    else:
        logging.info("Skipping weights loading, using randomly initialized model.")

    model = model.to(device)
    model.eval()

    agent = LearningAgent(model=model, encoder=encoder, device=device, temperature=0.0)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW puzzles AS SELECT * FROM '{args.data_path}'")

    brackets = list(range(args.min_elo, args.max_elo + 1, args.step_elo))
    empirical_win_rates = []
    actual_evaluated_elos = []

    logging.info(f"Starting Stratified Sampling. Brackets: {args.min_elo} to {args.max_elo} (Step {args.step_elo}).")

    for bracket in brackets:
        lower_bound = bracket - args.step_elo / 2.0
        upper_bound = bracket + args.step_elo / 2.0

        # Query random puzzles in this bracket
        query = f"""
        SELECT PuzzleId, FEN, Moves, Rating
        FROM puzzles
        WHERE Rating >= {lower_bound} AND Rating < {upper_bound}
        ORDER BY RANDOM()
        LIMIT {args.puzzles_per_bracket}
        """
        results = con.execute(query).fetchall()

        if not results:
            logging.warning(f"No puzzles found for bracket {bracket}. Skipping.")
            continue

        successes = 0
        for puzzle_id, fen, moves, rating in results:
            outcome = evaluate_puzzle(agent, fen, moves)
            successes += outcome

        win_rate = successes / len(results)
        empirical_win_rates.append(win_rate)
        actual_evaluated_elos.append(bracket)

        logging.info(f"Bracket {bracket}: Win Rate {win_rate * 100:.1f}% ({successes}/{len(results)} puzzles)")

    if len(empirical_win_rates) < 2:
        logging.error("Not enough data points collected to fit a curve.")
        return

    # Logistic Curve Fitting
    x_data = np.array(actual_evaluated_elos)
    y_data = np.array(empirical_win_rates)

    if np.max(y_data) == 0.0:
        logging.error(
            "Model win rate is 0.0% across all brackets. Cannot fit a meaningful curve. Elo is likely < min_elo."
        )
        estimated_elo = 0
        popt = None
    elif np.min(y_data) == 1.0:
        logging.error(
            "Model win rate is 100.0% across all brackets. Cannot fit a meaningful curve. Elo is likely > max_elo."
        )
        estimated_elo = 3000
        popt = None
    else:
        try:
            # Initial guess: k=0.01, x0=1500
            popt, _ = optimize.curve_fit(logistic_function, x_data, y_data, p0=[-0.01, 1500])
            estimated_elo = popt[1]
            k_val = popt[0]
            logging.info("=== Final Result ===")
            logging.info(f"Estimated 50% Puzzle Elo: {estimated_elo:.1f}")
            logging.info(f"Logistic falloff param k:  {k_val:.4f}")
        except RuntimeError:
            logging.error("Failed to fit logistic curve. The model performance may not follow a standard dropoff.")
            estimated_elo = 1500
            popt = None

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color="blue", label="Empirical Win Rate")

    if popt is not None:
        x_smooth = np.linspace(min(x_data), max(x_data), 200)
        y_smooth = logistic_function(x_smooth, *popt)
        plt.plot(x_smooth, y_smooth, color="red", label="Fitted Logistic Curve")
        plt.axvline(x=estimated_elo, color="green", linestyle="--", label=f"Estimated Elo: {estimated_elo:.0f}")
        plt.axhline(y=0.5, color="gray", linestyle=":")

    plt.title(f"Model Puzzle Solving Capability ({args.backbone})")
    plt.xlabel("Lichess Puzzle Elo")
    plt.ylabel("Win Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    Path("runs").mkdir(exist_ok=True)
    plot_path = "runs/elo_evaluation.png"
    plt.savefig(plot_path)
    logging.info(f"Saved evaluation plot to {plot_path}")


if __name__ == "__main__":
    main()
