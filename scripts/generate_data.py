#!/usr/bin/env python3
"""
Script to generate training data using bot-vs-bot games.

Supports both sequential and parallel generation modes.
"""
import argparse
from pathlib import Path

from src.config import load_config
from src.data import ChessDatabase, MatchRunner, ParallelMatchRunner


def main():
    parser = argparse.ArgumentParser(description="Generate chess training data")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Number of games to generate (overrides config)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Engine search depth (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output database path (overrides config)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel game generation using multiprocessing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get parameters (CLI overrides config)
    num_games = args.num_games or config.data_generation.num_games
    white_depth = getattr(config.data_generation, 'white_depth', 15)
    black_depth = getattr(config.data_generation, 'black_depth', 10)
    # CLI --depth overrides both if specified
    if args.depth:
        white_depth = args.depth
        black_depth = args.depth
    db_path = args.output or config.paths.database
    engine_path = config.engines['stockfish'].path
    openings_path = config.paths.openings
    multipv = config.engines['stockfish'].default_multipv
    
    # Get storage optimization settings with defaults
    skip_first_ply = getattr(config.data_generation, 'skip_first_ply', 8)
    sample_rate = getattr(config.data_generation, 'sample_rate', 2)
    
    print(f"Chess Training Data Generator")
    print(f"=" * 40)
    print(f"Engine: {engine_path}")
    print(f"Depth: white={white_depth}, black={black_depth}")
    print(f"MultiPV: {multipv}")
    print(f"Games: {num_games}")
    print(f"Database: {db_path}")
    print(f"Openings: {openings_path}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        print(f"Workers: {args.workers}")
    print(f"Storage Optimization: skip_first_ply={skip_first_ply}, sample_rate={sample_rate}")
    print()
    
    # Create database
    db = ChessDatabase(db_path)
    
    # Choose runner based on mode
    if args.parallel:
        runner = ParallelMatchRunner(
            database=db,
            openings_path=openings_path,
            max_moves=config.data_generation.max_moves_per_game,
            multipv=multipv,
            num_workers=args.workers,
            skip_first_ply=skip_first_ply,
            sample_rate=sample_rate,
        )
    else:
        runner = MatchRunner(
            database=db,
            openings_path=openings_path,
            max_moves=config.data_generation.max_moves_per_game,
            multipv=multipv,
            skip_first_ply=skip_first_ply,
            sample_rate=sample_rate,
        )
    
    # Generate games
    print("Starting data generation...")
    if args.parallel:
        stats = runner.run_games(
            engine_path=engine_path,
            num_games=num_games,
            white_depth=white_depth,
            black_depth=black_depth,
            save_every=config.data_generation.save_every,
            verbose=args.verbose,
        )
    else:
        # MatchRunner uses a single depth parameter
        stats = runner.run_games(
            engine_path=engine_path,
            num_games=num_games,
            depth=white_depth,  # Use white_depth as the main depth
            save_every=config.data_generation.save_every,
            verbose=args.verbose,
        )
    
    print()
    print(f"Generation Complete!")
    print(f"=" * 40)
    print(f"Games generated: {stats['num_games']}")
    print(f"Positions saved: {stats['total_positions']}")
    print(f"Results: {stats['results']}")
    print(f"Database size: {Path(db_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    db.close()


if __name__ == "__main__":
    main()

