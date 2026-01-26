#!/usr/bin/env python3
"""
Script to import positions from PGN files into the training database.

Supports:
- Large PGN files (streaming parser)
- Position sampling (every Nth position)
- Configurable analysis depth
- Progress tracking with ETA
- Resume capability (skip existing games based on hash)
"""
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Iterator
from tqdm import tqdm

import chess
import chess.pgn

from src.config import load_config
from src.data import ChessDatabase
from src.agents.uci_agent import UCIAgent


def parse_pgn_games(
    pgn_path: str | Path,
    max_games: Optional[int] = None,
) -> Iterator[chess.pgn.Game]:
    """
    Stream games from a PGN file.
    
    Args:
        pgn_path: Path to PGN file.
        max_games: Maximum number of games to yield.
    
    Yields:
        chess.pgn.Game objects.
    """
    count = 0
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            count += 1
            yield game
            
            if max_games is not None and count >= max_games:
                break


def game_hash(game: chess.pgn.Game) -> str:
    """Generate a unique hash for a game based on headers and moves."""
    data = ""
    
    # Include key headers
    for header in ['Event', 'Site', 'Date', 'White', 'Black', 'Result']:
        data += game.headers.get(header, '') + '|'
    
    # Include first 20 moves
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if i >= 20:
            break
        data += move.uci()
    
    return hashlib.md5(data.encode()).hexdigest()[:16]


def extract_positions(
    game: chess.pgn.Game,
    sample_rate: int = 1,
    skip_first: int = 0,
    skip_last: int = 0,
) -> list[dict]:
    """
    Extract positions from a game.
    
    Args:
        game: Chess game to extract from.
        sample_rate: Extract every Nth position (1 = all, 2 = every other, etc.)
        skip_first: Skip first N positions (avoid pure opening theory).
        skip_last: Skip last N positions (avoid trivial endgames).
    
    Returns:
        List of position dicts with 'fen' and 'ply'.
    """
    positions = []
    board = game.board()
    
    # Collect all positions first
    all_positions = [{'fen': board.fen(), 'ply': 0}]
    
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        all_positions.append({'fen': board.fen(), 'ply': ply})
    
    # Apply skip_first and skip_last
    if skip_last > 0:
        all_positions = all_positions[skip_first:-skip_last]
    else:
        all_positions = all_positions[skip_first:]
    
    # Apply sample_rate
    for i, pos in enumerate(all_positions):
        if i % sample_rate == 0:
            positions.append(pos)
    
    return positions


def analyze_and_store_game(
    game: chess.pgn.Game,
    database: ChessDatabase,
    analyzer: UCIAgent,
    sample_rate: int = 1,
    skip_first: int = 4,
    skip_last: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Analyze a game and store positions to database.
    
    Args:
        game: Chess game to analyze.
        database: Database to store positions.
        analyzer: UCI engine for analysis.
        sample_rate: Extract every Nth position.
        skip_first: Skip first N positions.
        skip_last: Skip last N positions.
        verbose: Print detailed output.
    
    Returns:
        Dict with number of positions stored.
    """
    # Get game metadata
    result = game.headers.get('Result', '*')
    white = game.headers.get('White', 'Unknown')
    black = game.headers.get('Black', 'Unknown')
    event = game.headers.get('Event', 'Unknown')
    opening = game.headers.get('Opening', event)
    
    # Get initial position (for opening_fen)
    board = game.board()
    opening_fen = board.fen()
    
    # Extract positions
    positions = extract_positions(
        game,
        sample_rate=sample_rate,
        skip_first=skip_first,
        skip_last=skip_last,
    )
    
    if not positions:
        return {'positions_stored': 0}
    
    # Count total moves for game record
    total_moves = sum(1 for _ in game.mainline_moves())
    
    # Add game to database
    game_id = database.add_game(
        opening_fen=opening_fen,
        opening_name=opening,
        result=result,
        white_agent=white,
        black_agent=black,
        num_moves=total_moves,
    )
    
    # Analyze each position and store
    positions_stored = 0
    for pos in positions:
        board = chess.Board(pos['fen'])
        
        # Skip positions where game is over
        if board.is_game_over():
            continue
        
        # Get move distribution from analyzer
        try:
            move_dist = analyzer.get_move_distribution(board)
        except Exception as e:
            if verbose:
                print(f"Analysis error at ply {pos['ply']}: {e}")
            continue
        
        # Store position with distribution
        database.add_position_with_distribution(
            game_id=game_id,
            fen=pos['fen'],
            ply=pos['ply'],
            move_distribution=move_dist,
        )
        
        positions_stored += 1
    
    return {'positions_stored': positions_stored}


def count_games_in_pgn(pgn_path: str | Path) -> int:
    """Quick count of games in a PGN file."""
    count = 0
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('[Event '):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Import positions from PGN files into training database"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to PGN file to import"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/imported.db",
        help="Output database path (default: data/imported.db)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=15,
        help="Engine analysis depth (default: 15)"
    )
    parser.add_argument(
        "--multipv",
        type=int,
        default=5,
        help="Number of moves per position (default: 5)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Extract every Nth position (default: 1 = all)"
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=4,
        help="Skip first N positions of each game (default: 4)"
    )
    parser.add_argument(
        "--skip-last",
        type=int,
        default=4,
        help="Skip last N positions of each game (default: 4)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to import"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    pgn_path = Path(args.input)
    if not pgn_path.exists():
        print(f"Error: PGN file not found: {pgn_path}")
        return 1
    
    # Load configuration
    config = load_config(args.config)
    engine_path = config.engines['stockfish'].path
    
    print("PGN Import Tool")
    print("=" * 40)
    print(f"Input: {pgn_path}")
    print(f"Output: {args.output}")
    print(f"Engine: {engine_path}")
    print(f"Depth: {args.depth}")
    print(f"MultiPV: {args.multipv}")
    print(f"Sample rate: every {args.sample_rate} position(s)")
    print(f"Skip: first {args.skip_first}, last {args.skip_last}")
    if args.max_games:
        print(f"Max games: {args.max_games}")
    print()
    
    # Count games for progress bar
    print("Counting games in PGN file...")
    total_games = count_games_in_pgn(pgn_path)
    if args.max_games:
        total_games = min(total_games, args.max_games)
    print(f"Found {total_games} games to process")
    print()
    
    # Create database
    db = ChessDatabase(args.output)
    
    # Create analyzer engine
    total_positions = 0
    games_processed = 0
    
    print("Starting import...")
    
    with UCIAgent(engine_path, depth=args.depth, multipv=args.multipv) as analyzer:
        games = parse_pgn_games(pgn_path, max_games=args.max_games)
        
        for game in tqdm(games, total=total_games, desc="Importing games"):
            try:
                result = analyze_and_store_game(
                    game=game,
                    database=db,
                    analyzer=analyzer,
                    sample_rate=args.sample_rate,
                    skip_first=args.skip_first,
                    skip_last=args.skip_last,
                    verbose=args.verbose,
                )
                
                total_positions += result['positions_stored']
                games_processed += 1
                
            except Exception as e:
                if args.verbose:
                    print(f"\nError processing game: {e}")
                continue
    
    print()
    print("Import Complete!")
    print("=" * 40)
    print(f"Games imported: {games_processed}")
    print(f"Positions stored: {total_positions}")
    print(f"Database: {args.output}")
    
    db_path = Path(args.output)
    if db_path.exists():
        print(f"Database size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    db.close()
    return 0


if __name__ == "__main__":
    exit(main())
