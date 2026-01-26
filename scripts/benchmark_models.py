#!/usr/bin/env python3
"""
Benchmark trained models against Stockfish at various depths to estimate Elo.

Stockfish depth to approximate Elo mapping (rough estimates):
- Depth 1:  ~800 Elo
- Depth 2:  ~1000 Elo
- Depth 3:  ~1200 Elo
- Depth 5:  ~1500 Elo
- Depth 8:  ~1800 Elo
- Depth 10: ~2000 Elo
- Depth 12: ~2200 Elo
- Depth 15: ~2500 Elo
- Depth 20: ~2800 Elo
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import math
import time

import chess
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.agents.learning_agent import LearningAgent
from src.agents.uci_agent import UCIAgent
from src.device import get_device


# Approximate Elo for Stockfish at each depth (empirically derived estimates)
STOCKFISH_DEPTH_ELO = {
    1: 800,
    2: 1000,
    3: 1200,
    4: 1350,
    5: 1500,
    6: 1650,
    7: 1750,
    8: 1850,
    9: 1950,
    10: 2050,
    12: 2250,
    15: 2500,
    20: 2800,
}

# Models to benchmark
TRAINED_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer"]


def load_model_agent(model_name: str, checkpoint_dir: Path, device: torch.device) -> LearningAgent:
    """Load a trained model and create a LearningAgent."""
    # Load config
    config = load_config("config/default.yaml")
    config.model.backbone = model_name
    config.model.head = "dual"
    
    # Create model
    model = create_model(config.model)
    
    # Load checkpoint
    checkpoint_path = checkpoint_dir / model_name / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get encoder
    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()
    
    # Create agent
    agent = LearningAgent(
        model=model,
        encoder=encoder,
        device=device,
        temperature=0.0,  # Greedy selection for benchmarking
    )
    
    return agent


def play_game(white_agent, black_agent, max_moves: int = 200) -> dict:
    """
    Play a single game between two agents.
    
    Returns:
        Dict with 'result' (1.0, 0.5, 0.0 for white win/draw/loss),
        'moves', 'reason'.
    """
    board = chess.Board()
    moves = []
    
    while not board.is_game_over() and len(moves) < max_moves:
        if board.turn == chess.WHITE:
            move = white_agent.get_move(board)
        else:
            move = black_agent.get_move(board)
        
        board.push(move)
        moves.append(move.uci())
    
    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = 0.0  # Black (who just moved) wins
            reason = "checkmate"
        else:
            result = 1.0  # White wins
            reason = "checkmate"
    elif board.is_stalemate():
        result = 0.5
        reason = "stalemate"
    elif board.is_insufficient_material():
        result = 0.5
        reason = "insufficient_material"
    elif board.is_fifty_moves():
        result = 0.5
        reason = "fifty_moves"
    elif board.is_repetition():
        result = 0.5
        reason = "repetition"
    elif len(moves) >= max_moves:
        result = 0.5
        reason = "max_moves"
    else:
        result = 0.5
        reason = "unknown"
    
    return {
        'result': result,
        'moves': len(moves),
        'reason': reason,
    }


def benchmark_against_stockfish(
    agent: LearningAgent,
    stockfish_path: str,
    depth: int,
    num_games: int = 10,
) -> dict:
    """
    Benchmark an agent against Stockfish at a specific depth.
    
    Returns:
        Dict with win/loss/draw counts and statistics.
    """
    stockfish = UCIAgent(stockfish_path, depth=depth)
    
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    
    try:
        # Play games alternating colors
        for i in tqdm(range(num_games), desc=f"Depth {depth}"):
            if i % 2 == 0:
                # Agent plays white
                game = play_game(agent, stockfish)
                if game['result'] == 1.0:
                    wins += 1
                elif game['result'] == 0.0:
                    losses += 1
                else:
                    draws += 1
            else:
                # Agent plays black
                game = play_game(stockfish, agent)
                if game['result'] == 0.0:
                    wins += 1
                elif game['result'] == 1.0:
                    losses += 1
                else:
                    draws += 1
            
            total_moves += game['moves']
    finally:
        stockfish.close()
    
    score = wins + 0.5 * draws
    total = wins + losses + draws
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'score': score,
        'total': total,
        'win_rate': wins / total if total > 0 else 0,
        'score_rate': score / total if total > 0 else 0.5,
        'avg_moves': total_moves / total if total > 0 else 0,
    }


def estimate_elo_from_score(score_rate: float, opponent_elo: int) -> int:
    """
    Estimate Elo based on score rate against a known opponent.
    
    Uses the Elo expected score formula inverted.
    """
    if score_rate >= 1.0:
        return opponent_elo + 400  # Cap at +400
    elif score_rate <= 0.0:
        return opponent_elo - 400  # Cap at -400
    
    # Invert the expected score formula
    # E = 1 / (1 + 10^((Rb - Ra)/400))
    # Solving for Ra: Ra = Rb - 400 * log10(1/E - 1)
    elo_diff = -400 * math.log10(1 / score_rate - 1)
    return int(opponent_elo + elo_diff)


def find_crossover_elo(depth_results: dict) -> int:
    """
    Find Elo where model achieves ~50% score (crossover point).
    Uses linear interpolation between tested depths.
    """
    depths = sorted(depth_results.keys())
    
    # Find where score crosses 0.5
    for i in range(len(depths) - 1):
        d1, d2 = depths[i], depths[i + 1]
        s1 = depth_results[d1]['score_rate']
        s2 = depth_results[d2]['score_rate']
        
        if s1 >= 0.5 >= s2:
            # Linear interpolation
            ratio = (s1 - 0.5) / (s1 - s2) if s1 != s2 else 0.5
            elo1 = STOCKFISH_DEPTH_ELO.get(d1, d1 * 150 + 600)
            elo2 = STOCKFISH_DEPTH_ELO.get(d2, d2 * 150 + 600)
            return int(elo1 + ratio * (elo2 - elo1))
    
    # If no crossover, estimate from best/worst performance
    best_depth = max(depths)
    best_score = depth_results[best_depth]['score_rate']
    best_elo = STOCKFISH_DEPTH_ELO.get(best_depth, best_depth * 150 + 600)
    
    return estimate_elo_from_score(best_score, best_elo)


def run_benchmark(
    checkpoint_dir: Path,
    stockfish_path: str,
    depths: list[int],
    num_games: int,
    output_dir: Path,
):
    """Run full benchmark for all models."""
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name in TRAINED_MODELS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            agent = load_model_agent(model_name, checkpoint_dir, device)
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")
            continue
        
        model_results = {}
        
        for depth in depths:
            print(f"\nPlaying {num_games} games vs Stockfish depth {depth}...")
            result = benchmark_against_stockfish(
                agent=agent,
                stockfish_path=stockfish_path,
                depth=depth,
                num_games=num_games,
            )
            model_results[depth] = result
            
            stockfish_elo = STOCKFISH_DEPTH_ELO.get(depth, depth * 150 + 600)
            print(f"  Wins: {result['wins']}, Draws: {result['draws']}, Losses: {result['losses']}")
            print(f"  Score Rate: {result['score_rate']:.1%} vs ~{stockfish_elo} Elo")
        
        # Estimate Elo
        estimated_elo = find_crossover_elo(model_results)
        
        all_results[model_name] = {
            'depth_results': {str(k): v for k, v in model_results.items()},
            'estimated_elo': estimated_elo,
        }
        
        print(f"\n  Estimated Elo: {estimated_elo}")
    
    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    generate_benchmark_visualizations(all_results, depths, output_dir)
    
    # Generate report
    generate_benchmark_report(all_results, depths, output_dir)
    
    return all_results


def generate_benchmark_visualizations(results: dict, depths: list[int], output_dir: Path):
    """Generate benchmark visualizations."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    models = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 1. Score Rate vs Stockfish Depth
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (model_name, data) in enumerate(results.items()):
        depth_results = data['depth_results']
        x = sorted([int(d) for d in depth_results.keys()])
        y = [depth_results[str(d)]['score_rate'] for d in x]
        ax.plot(x, y, 'o-', label=model_name, color=colors[i], linewidth=2, markersize=8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Score Line')
    ax.set_xlabel('Stockfish Depth', fontsize=12)
    ax.set_ylabel('Score Rate', fontsize=12)
    ax.set_title('Model Performance vs Stockfish Depth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add secondary x-axis for approximate Elo
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    elo_ticks = [STOCKFISH_DEPTH_ELO.get(d, d * 150 + 600) for d in depths]
    ax2.set_xticks(depths)
    ax2.set_xticklabels([f"~{e}" for e in elo_ticks])
    ax2.set_xlabel('Approx. Stockfish Elo', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'score_vs_depth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'score_vs_depth.png'}")
    
    # 2. Estimated Elo Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    elos = [data['estimated_elo'] for data in results.values()]
    x = np.arange(len(models))
    bars = ax.bar(x, elos, color=colors[:len(models)])
    
    # Add value labels
    for bar, elo in zip(bars, elos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                f'{elo}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Estimated Elo', fontsize=12)
    ax.set_title('Estimated Elo Rating by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'estimated_elo.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'estimated_elo.png'}")
    
    # 3. Win/Draw/Loss breakdown
    fig, axes = plt.subplots(1, len(depths), figsize=(4*len(depths), 5))
    if len(depths) == 1:
        axes = [axes]
    
    for ax, depth in zip(axes, sorted(depths)):
        wins = [results[m]['depth_results'][str(depth)]['wins'] for m in models]
        draws = [results[m]['depth_results'][str(depth)]['draws'] for m in models]
        losses = [results[m]['depth_results'][str(depth)]['losses'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, wins, width, label='Wins', color='green')
        ax.bar(x, draws, width, label='Draws', color='gray')
        ax.bar(x + width, losses, width, label='Losses', color='red')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Count')
        ax.set_title(f'Depth {depth}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
    
    plt.suptitle('Win/Draw/Loss Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'wdl_breakdown.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'wdl_breakdown.png'}")


def generate_benchmark_report(results: dict, depths: list[int], output_dir: Path):
    """Generate benchmark report."""
    report_path = output_dir / "benchmark_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Chess Model Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Estimated Elo Ratings\n\n")
        f.write("| Model | Estimated Elo | Rank |\n")
        f.write("|-------|---------------|------|\n")
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['estimated_elo'], reverse=True)
        for rank, (model, data) in enumerate(sorted_models, 1):
            f.write(f"| {model} | **{data['estimated_elo']}** | #{rank} |\n")
        
        f.write("\n## Performance vs Stockfish Depth\n\n")
        
        header = "| Model |" + "|".join([f" D{d} |" for d in sorted(depths)])
        f.write(header + "\n")
        f.write("|" + "---|" * (len(depths) + 1) + "\n")
        
        for model, data in results.items():
            row = f"| {model} |"
            for d in sorted(depths):
                sr = data['depth_results'][str(d)]['score_rate']
                row += f" {sr:.1%} |"
            f.write(row + "\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for model, data in results.items():
            f.write(f"### {model}\n\n")
            f.write(f"**Estimated Elo:** {data['estimated_elo']}\n\n")
            f.write("| Depth | SF Elo | W | D | L | Score | Rate |\n")
            f.write("|-------|--------|---|---|---|-------|------|\n")
            
            for d in sorted([int(k) for k in data['depth_results'].keys()]):
                res = data['depth_results'][str(d)]
                sf_elo = STOCKFISH_DEPTH_ELO.get(d, d * 150 + 600)
                f.write(f"| {d} | ~{sf_elo} | {res['wins']} | {res['draws']} | {res['losses']} | {res['score']:.1f}/{res['total']} | {res['score_rate']:.1%} |\n")
            f.write("\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Score vs Depth](figures/score_vs_depth.png)\n\n")
        f.write("![Estimated Elo](figures/estimated_elo.png)\n\n")
        f.write("![WDL Breakdown](figures/wdl_breakdown.png)\n\n")
        
        f.write("## Methodology\n\n")
        f.write("- Models play alternating colors (white/black) for fairness\n")
        f.write("- Greedy move selection (temperature=0) for deterministic play\n")
        f.write("- Games capped at 200 moves\n")
        f.write("- Elo estimated from crossover point (50% score rate)\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark models against Stockfish")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="training_results",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 3, 5, 8],
        help="Stockfish depths to benchmark against"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games per depth (will be split between colors)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHESS MODEL BENCHMARK vs STOCKFISH")
    print("=" * 60)
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Stockfish: {args.stockfish}")
    print(f"Depths: {args.depths}")
    print(f"Games per depth: {args.games}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    run_benchmark(
        checkpoint_dir=Path(args.checkpoint_dir),
        stockfish_path=args.stockfish,
        depths=args.depths,
        num_games=args.games,
        output_dir=Path(args.output_dir),
    )
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
