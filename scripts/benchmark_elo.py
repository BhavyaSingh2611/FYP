"""
Elo-based Benchmark using Lichess Puzzles and Computerized Adaptive Testing (CAT).
"""

import argparse
import logging
import math
from pathlib import Path

import chess
import duckdb
import torch
import numpy as np
from scipy import optimize

from src.agents.learning_agent import LearningAgent
from src.models.factory import create_model, get_encoder_for_model
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


DUMMY_PRIOR_GAMES = [
    (1500, 1),
    (1500, 0),
]


def prob_solve(rating: float, puzzle_rating: float) -> float:
    """Expected probability of solving a puzzle of puzzle_rating for a model of rating."""
    return 1.0 / (1.0 + 10.0 ** ((puzzle_rating - rating) / 400.0))


def estimate_elo_map(outcomes: list[dict], prior_mu: float = 1500, prior_sigma: float = 400) -> tuple[float, float]:
    """
    Maximum A Posteriori (MAP) estimation of Elo.
    
    Args:
        outcomes: List of dicts with 'rating' and 'outcome' (1 for solve, 0 for fail).
        prior_mu: Prior mean Elo.
        prior_sigma: Prior standard deviation.
        
    Returns:
        tuple of (estimated_elo, standard_error).
    """
    # Augment with dummy games if it is totally empty, though the Gaussian prior stabilizes it.
    all_games = [{"rating": r, "outcome": o} for r, o in DUMMY_PRIOR_GAMES] + outcomes

    def negative_log_posterior(theta):
        # Log likelihood
        ll = 0.0
        for game in all_games:
            p = prob_solve(theta, game["rating"])
            # Clip p to prevent log(0)
            p = max(1e-9, min(1.0 - 1e-9, p))
            if game["outcome"] == 1:
                ll += math.log(p)
            else:
                ll += math.log(1 - p)
        
        # Gaussian prior
        prior = -0.5 * ((theta - prior_mu) / prior_sigma) ** 2
        return -(ll + prior)

    # Optimization
    res = optimize.minimize_scalar(negative_log_posterior, bounds=(0, 4000), method='bounded')
    est_elo = res.x

    # Calculate Fisher Information at the estimate to get standard error
    fisher_info = 1.0 / (prior_sigma ** 2)
    for game in all_games:
        p = prob_solve(est_elo, game["rating"])
        # Derivative of log likelihood w.r.t theta
        # dp/dtheta = p(1-p) * ln(10) / 400
        # fisher info for one observation: pq [ln(10)/400]^2
        fisher_info += p * (1 - p) * ((math.log(10) / 400.0) ** 2)

    se = 1.0 / math.sqrt(fisher_info)
    return est_elo, se


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
    parser = argparse.ArgumentParser(description="Evaluate a chess model's Elo using Lichess Puzzles via CAT.")
    parser.add_argument("--backbone", type=str, default="resnet", help="Network backbone architecture")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt). If None, uses randomly initialized weights.")
    parser.add_argument("--data-path", type=str, default="data/puzzles/*.parquet", help="Path or glob pattern to Lichess puzzle parquet files.")
    parser.add_argument("--target-ci", type=float, default=100.0, help="Target 95%% Confidence Interval width (e.g. 100 means +/- 50).")
    parser.add_argument("--max-puzzles", type=int, default=300, help="Maximum number of puzzles to evaluate before stopping.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
                cleaned_state_dict[k[len("_orig_mod."):]] = v
            else:
                cleaned_state_dict[k] = v
                
        model.load_state_dict(cleaned_state_dict)
        logging.info(f"Loaded weights from {args.weights}")
    else:
        logging.info("Skipping weights loading, using randomly initialized model.")

    model = model.to(device)
    model.eval()

    # Create Agent
    # For a puzzle benchmark, greedy (temperature=0) choice is ideal unless we want to take multiple samples.
    agent = LearningAgent(model=model, encoder=encoder, device=device, temperature=0.0)

    # Initialize CAT
    current_elo = 1500.0
    current_se = 400.0
    outcomes = []
    
    con = duckdb.connect()
    con.execute(f"CREATE VIEW puzzles AS SELECT * FROM '{args.data_path}'")
    
    # Track used puzzle IDs to prevent repeats
    used_puzzle_ids = set()

    logging.info(f"Starting Elo estimation. Target 95% CI width: {args.target_ci}")

    for step in range(args.max_puzzles):
        # We want a puzzle close to current_elo
        bracket_width = 50
        
        while True:
            # Query random puzzle roughly around current_elo +/- bracket_width
            query = f"""
            SELECT PuzzleId, FEN, Moves, Rating
            FROM puzzles
            WHERE Rating BETWEEN {current_elo - bracket_width} AND {current_elo + bracket_width}
            ORDER BY RANDOM()
            LIMIT 1
            """
            result = con.execute(query).fetchone()
            
            if result:
                puzzle_id, fen, moves, rating = result
                if puzzle_id not in used_puzzle_ids:
                    used_puzzle_ids.add(puzzle_id)
                    break
            
            # Widen bracket if no unplayed puzzle was found
            bracket_width += 50
            if bracket_width > 1000:
                logging.error("Exhausted puzzles in database! Stopping early.")
                return

        # Evaluate puzzle
        outcome = evaluate_puzzle(agent, fen, moves)
        outcomes.append({"rating": rating, "outcome": outcome})
        
        # Update Estimation
        new_elo, new_se = estimate_elo_map(outcomes)
        ci_width = 2 * 1.96 * new_se
        
        current_elo = new_elo
        current_se = new_se
        
        logging.info(f"Step {step+1}: Played Puzzle {puzzle_id} (Elo {rating}) | Outcome: {'WIN' if outcome else 'LOSS'} | Current Est: {current_elo:.1f} ± {(ci_width/2):.1f} (Target: ±{args.target_ci/2:.1f})")
        
        if ci_width <= args.target_ci:
            logging.info(f"Reached Target Confidence Interval! Stopping at {step+1} puzzles.")
            break

    logging.info(f"=== Final Result ===")
    logging.info(f"Puzzles Played: {len(outcomes)}")
    logging.info(f"Estimated Elo:  {current_elo:.2f}")
    logging.info(f"95% CI bounds:  [{current_elo - 1.96*current_se:.2f}, {current_elo + 1.96*current_se:.2f}] (width: {2*1.96*current_se:.2f})")


if __name__ == "__main__":
    main()
