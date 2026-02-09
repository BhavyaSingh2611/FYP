#!/usr/bin/env python3
"""
Detailed benchmark with move-by-move analysis.

Features:
- Records all moves played in each game
- Tracks centipawn evaluation (using Stockfish depth 18) for each position
- Generates evaluation flow graphs for each game
- Exports games in PGN format for chess software preview
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

import chess
import chess.pgn
import chess.engine
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.agents.learning_agent import LearningAgent
from src.agents.uci_agent import UCIAgent
from src.device import get_device


# Models to benchmark
TRAINED_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]

# Evaluation depth for centipawn tracking
EVAL_DEPTH = 18


def load_model_agent(
    model_name: str, 
    checkpoint_dir: Path, 
    device: torch.device,
    checkpoint_path: Path = None,
) -> LearningAgent:
    """Load a trained model and create a LearningAgent."""
    config = load_config("config/default.yaml")
    config.model.backbone = model_name
    config.model.head = "dual"
    
    model = create_model(config.model)
    
    if checkpoint_path is None:
        checkpoint_path = checkpoint_dir / model_name / "final.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()
    
    agent = LearningAgent(
        model=model,
        encoder=encoder,
        device=device,
        temperature=0.0,
    )
    
    return agent


def evaluate_position(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int = EVAL_DEPTH) -> int:
    """
    Evaluate a position using Stockfish.
    
    Returns centipawn score from white's perspective.
    Mate scores are converted to large values (±10000).
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info['score'].white()
        
        if score.is_mate():
            mate_in = score.mate()
            if mate_in > 0:
                return 10000 - mate_in * 10  # Mate in N moves
            else:
                return -10000 - mate_in * 10  # Getting mated
        else:
            return score.score()
    except Exception as e:
        return 0


def play_game_with_analysis(
    white_agent,
    black_agent,
    evaluator: chess.engine.SimpleEngine,
    max_moves: int = 200,
    eval_depth: int = EVAL_DEPTH,
) -> dict:
    """
    Play a game with full move and evaluation tracking.
    
    Returns:
        Dict with moves, evaluations, result, and metadata.
    """
    board = chess.Board()
    game_data = {
        'moves': [],           # UCI moves
        'san_moves': [],       # Standard algebraic notation
        'evaluations': [],     # Centipawn after each move
        'move_times': [],      # Time taken for each move
        'white_agent': getattr(white_agent, 'name', str(white_agent)),
        'black_agent': getattr(black_agent, 'name', str(black_agent)),
    }
    
    # Initial evaluation
    initial_eval = evaluate_position(evaluator, board, eval_depth)
    game_data['evaluations'].append(initial_eval)
    
    while not board.is_game_over() and len(game_data['moves']) < max_moves:
        start_time = time.time()
        
        # Get move from current player
        if board.turn == chess.WHITE:
            move = white_agent.get_move(board)
        else:
            move = black_agent.get_move(board)
        
        move_time = time.time() - start_time
        
        # Record move in both formats
        san = board.san(move)
        game_data['moves'].append(move.uci())
        game_data['san_moves'].append(san)
        game_data['move_times'].append(move_time)
        
        # Make the move
        board.push(move)
        
        # Evaluate new position
        eval_score = evaluate_position(evaluator, board, eval_depth)
        game_data['evaluations'].append(eval_score)
    
    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            game_data['result'] = '0-1'
            game_data['termination'] = 'checkmate'
        else:
            game_data['result'] = '1-0'
            game_data['termination'] = 'checkmate'
    elif board.is_stalemate():
        game_data['result'] = '1/2-1/2'
        game_data['termination'] = 'stalemate'
    elif board.is_insufficient_material():
        game_data['result'] = '1/2-1/2'
        game_data['termination'] = 'insufficient_material'
    elif board.is_fifty_moves():
        game_data['result'] = '1/2-1/2'
        game_data['termination'] = 'fifty_moves'
    elif board.is_repetition():
        game_data['result'] = '1/2-1/2'
        game_data['termination'] = 'repetition'
    elif len(game_data['moves']) >= max_moves:
        game_data['result'] = '1/2-1/2'
        game_data['termination'] = 'max_moves'
    else:
        game_data['result'] = '*'
        game_data['termination'] = 'unknown'
    
    game_data['final_fen'] = board.fen()
    game_data['total_moves'] = len(game_data['moves'])
    
    return game_data


def create_pgn(game_data: dict, event: str = "Model Benchmark") -> chess.pgn.Game:
    """Create a PGN game from game data."""
    game = chess.pgn.Game()
    
    # Set headers
    game.headers["Event"] = event
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = game_data['white_agent']
    game.headers["Black"] = game_data['black_agent']
    game.headers["Result"] = game_data['result']
    game.headers["Termination"] = game_data.get('termination', 'unknown')
    
    # Add moves
    node = game
    board = chess.Board()
    
    for i, uci_move in enumerate(game_data['moves']):
        move = chess.Move.from_uci(uci_move)
        node = node.add_variation(move)
        
        # Add evaluation as comment
        if i + 1 < len(game_data['evaluations']):
            eval_cp = game_data['evaluations'][i + 1]
            if abs(eval_cp) >= 10000:
                mate_in = (10000 - abs(eval_cp)) // 10
                if eval_cp > 0:
                    node.comment = f"[%eval #{mate_in}]"
                else:
                    node.comment = f"[%eval #-{mate_in}]"
            else:
                node.comment = f"[%eval {eval_cp / 100:.2f}]"
        
        board.push(move)
    
    return game


def plot_evaluation_flow(game_data: dict, output_path: Path, title: str = None):
    """Create evaluation flow chart for a single game."""
    evals = game_data['evaluations']
    moves = list(range(len(evals)))
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Convert to pawns for readability
    evals_pawns = [e / 100 for e in evals]
    
    # Clip extreme values for better visualization
    evals_clipped = np.clip(evals_pawns, -10, 10)
    
    # Fill areas
    ax.fill_between(moves, 0, evals_clipped, where=np.array(evals_clipped) > 0, 
                    color='white', alpha=0.8, label='White advantage')
    ax.fill_between(moves, 0, evals_clipped, where=np.array(evals_clipped) <= 0, 
                    color='black', alpha=0.6, label='Black advantage')
    
    # Plot line
    ax.plot(moves, evals_clipped, 'b-', linewidth=1.5, alpha=0.7)
    
    # Reference line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Move Number', fontsize=11)
    ax.set_ylabel('Evaluation (pawns)', fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f"{game_data['white_agent']} vs {game_data['black_agent']} - {game_data['result']}", 
                    fontsize=12, fontweight='bold')
    
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add result annotation
    result_color = 'green' if game_data['result'] == '1-0' else 'red' if game_data['result'] == '0-1' else 'gray'
    ax.annotate(f"Result: {game_data['result']}\n({game_data['termination']})", 
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_games_comparison(all_games: dict, output_dir: Path):
    """Create comparison plots for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax_idx, (model_name, games) in enumerate(all_games.items()):
        if ax_idx >= 4:
            break
        
        ax = axes[ax_idx]
        
        for i, game in enumerate(games):
            evals = [e / 100 for e in game['evaluations']]
            evals_clipped = np.clip(evals, -10, 10)
            
            # Determine line style based on model's color
            if model_name in game['white_agent']:
                linestyle = '-'
                label = f"Game {i+1} (as White)" if i < 3 else None
            else:
                linestyle = '--'
                label = f"Game {i+1} (as Black)" if i < 3 else None
            
            ax.plot(evals_clipped, linestyle=linestyle, alpha=0.7, linewidth=1.5, label=label)
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Move Number')
        ax.set_ylabel('Evaluation (pawns)')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('Evaluation Flow by Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_games_comparison.png', dpi=150)
    plt.close()


def run_detailed_benchmark(
    checkpoint_dir: Path,
    stockfish_path: str,
    opponent_depth: int,
    num_games: int,
    output_dir: Path,
    single_model: str = None,
    single_checkpoint: Path = None,
    skill_level: int = None,
):
    """Run detailed benchmark with full analysis."""
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    pgn_dir = output_dir / "pgn"
    pgn_dir.mkdir(exist_ok=True)
    
    # Create evaluator engine (high depth for accurate analysis)
    print(f"Starting Stockfish evaluator (depth {EVAL_DEPTH})...")
    evaluator = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    all_games = {}
    all_results = {}
    
    # If single model specified, only benchmark that one
    models_to_benchmark = [single_model] if single_model else TRAINED_MODELS
    
    try:
        for model_name in models_to_benchmark:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name.upper()}")
            print(f"{'='*60}")
            
            try:
                ckpt_path = single_checkpoint if single_model else None
                agent = load_model_agent(model_name, checkpoint_dir, device, ckpt_path)
            except FileNotFoundError as e:
                print(f"Skipping {model_name}: {e}")
                continue
            
            # Create opponent Stockfish
            opponent = UCIAgent(stockfish_path, depth=opponent_depth, skill_level=skill_level)
            
            model_games = []
            wins, draws, losses = 0, 0, 0
            
            try:
                for game_num in range(num_games):
                    print(f"\nGame {game_num + 1}/{num_games}...", end=" ")
                    
                    # Alternate colors
                    if game_num % 2 == 0:
                        game_data = play_game_with_analysis(
                            white_agent=agent,
                            black_agent=opponent,
                            evaluator=evaluator,
                            eval_depth=EVAL_DEPTH,
                        )
                        model_color = 'white'
                    else:
                        game_data = play_game_with_analysis(
                            white_agent=opponent,
                            black_agent=agent,
                            evaluator=evaluator,
                            eval_depth=EVAL_DEPTH,
                        )
                        model_color = 'black'
                    
                    game_data['model_color'] = model_color
                    game_data['game_number'] = game_num + 1
                    
                    # Track results
                    if game_data['result'] == '1-0':
                        if model_color == 'white':
                            wins += 1
                        else:
                            losses += 1
                    elif game_data['result'] == '0-1':
                        if model_color == 'black':
                            wins += 1
                        else:
                            losses += 1
                    else:
                        draws += 1
                    
                    print(f"{game_data['result']} ({game_data['total_moves']} moves, {game_data['termination']})")
                    
                    model_games.append(game_data)
                    
                    # Create PGN
                    pgn = create_pgn(game_data, f"Benchmark: {model_name} vs Stockfish D{opponent_depth}")
                    pgn_path = pgn_dir / f"{model_name}_game_{game_num + 1}.pgn"
                    with open(pgn_path, 'w') as f:
                        f.write(str(pgn))
                    
                    # Create evaluation plot
                    plot_path = figures_dir / f"{model_name}_game_{game_num + 1}_eval.png"
                    plot_evaluation_flow(
                        game_data, 
                        plot_path,
                        f"{model_name} Game {game_num + 1}: {game_data['result']}"
                    )
                
            finally:
                opponent.close()
            
            all_games[model_name] = model_games
            all_results[model_name] = {
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'games': len(model_games),
            }
            
            print(f"\nResults: {wins}W - {draws}D - {losses}L")
        
        # Create combined PGN file
        combined_pgn_path = output_dir / "all_games.pgn"
        with open(combined_pgn_path, 'w') as f:
            for model_name, games in all_games.items():
                for game_data in games:
                    pgn = create_pgn(game_data)
                    f.write(str(pgn))
                    f.write("\n\n")
        print(f"\nAll games saved to: {combined_pgn_path}")
        
        # Create comparison plot
        plot_all_games_comparison(all_games, figures_dir)
        
        # Save JSON data
        json_data = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'opponent_depth': opponent_depth,
                'eval_depth': EVAL_DEPTH,
                'games_per_model': num_games,
            },
            'results': all_results,
            'games': {
                model: [
                    {
                        'game_number': g['game_number'],
                        'model_color': g['model_color'],
                        'result': g['result'],
                        'termination': g['termination'],
                        'total_moves': g['total_moves'],
                        'moves': g['san_moves'],
                        'evaluations': g['evaluations'],
                    }
                    for g in games
                ]
                for model, games in all_games.items()
            }
        }
        
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Generate report
        generate_detailed_report(all_results, all_games, output_dir, opponent_depth)
        
    finally:
        evaluator.quit()
    
    return all_games, all_results


def generate_detailed_report(results: dict, games: dict, output_dir: Path, opponent_depth: int):
    """Generate detailed markdown report."""
    report_path = output_dir / "detailed_benchmark_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Detailed Chess Model Benchmark\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Opponent:** Stockfish depth {opponent_depth}\n")
        f.write(f"**Evaluation Depth:** {EVAL_DEPTH}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Model | W | D | L | Score |\n")
        f.write("|-------|---|---|---|-------|\n")
        
        for model, res in results.items():
            total = res['wins'] + res['draws'] + res['losses']
            score = res['wins'] + 0.5 * res['draws']
            f.write(f"| {model} | {res['wins']} | {res['draws']} | {res['losses']} | {score}/{total} |\n")
        
        f.write("\n## Game Files\n\n")
        f.write("- **Combined PGN:** [all_games.pgn](all_games.pgn) - Open in Lichess, Chess.com, or any chess software\n")
        f.write("- **Individual PGNs:** `pgn/` directory\n\n")
        
        f.write("## Evaluation Charts\n\n")
        f.write("![All Games Comparison](figures/all_games_comparison.png)\n\n")
        
        f.write("### Individual Game Charts\n\n")
        for model in games.keys():
            f.write(f"#### {model}\n\n")
            for i in range(len(games[model])):
                f.write(f"![Game {i+1}](figures/{model}_game_{i+1}_eval.png)\n\n")
        
        f.write("## How to View Games\n\n")
        f.write("1. **Lichess:** Go to lichess.org/paste and paste the PGN content\n")
        f.write("2. **Chess.com:** Use chess.com/analysis and import PGN\n")
        f.write("3. **Desktop Apps:** Open with ChessBase, Arena, SCID, or Lucas Chess\n")
        f.write("4. **Command Line:** Use `python-chess` to parse the PGN files\n")
    
    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Detailed benchmark with move analysis")
    parser.add_argument("--checkpoint-dir", type=str, default="training_results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file path")
    parser.add_argument("--model", type=str, default=None, 
                       choices=TRAINED_MODELS,
                       help="Single model to benchmark (required with --checkpoint)")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--opponent-depth", type=int, default=5, help="Stockfish opponent depth")
    parser.add_argument("--skill-level", type=int, default=None, help="Stockfish skill level (0-20, None for full strength)")
    parser.add_argument("--games", type=int, default=4, help="Games per model (even number)")
    parser.add_argument("--output-dir", type=str, default="detailed_benchmark")
    parser.add_argument("--name", type=str, default=None, help="Run name (saves to runs/<name>/benchmark/)")
    
    args = parser.parse_args()
    
    if args.name:
        args.output_dir = f"runs/{args.name}/benchmark"
        args.checkpoint_dir = f"runs/{args.name}/training"
    
    if args.checkpoint and not args.model:
        parser.error("--model is required when using --checkpoint")
    
    print("=" * 60)
    print("DETAILED CHESS MODEL BENCHMARK")
    if args.name:
        print(f"Run: {args.name}")
    print("=" * 60)
    print(f"Opponent: Stockfish depth {args.opponent_depth}" + (f", skill level {args.skill_level}" if args.skill_level is not None else ""))
    print(f"Evaluation: Stockfish depth {EVAL_DEPTH}")
    if args.model:
        print(f"Model: {args.model}")
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print(f"Models: All")
    print(f"Games per model: {args.games}")
    print("=" * 60)
    
    run_detailed_benchmark(
        checkpoint_dir=Path(args.checkpoint_dir),
        stockfish_path=args.stockfish,
        opponent_depth=args.opponent_depth,
        num_games=args.games,
        output_dir=Path(args.output_dir),
        single_model=args.model,
        single_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        skill_level=args.skill_level,
    )
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
