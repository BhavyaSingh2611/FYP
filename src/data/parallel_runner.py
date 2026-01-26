"""
Parallel match runner for faster training data generation using multiprocessing.
"""
import json
import random
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from functools import partial

import chess

from .database import ChessDatabase
from ..agents.uci_agent import UCIAgent


def _run_single_game_worker(
    game_args: dict,
    engine_path: str,
    white_depth: int,
    black_depth: int,
    multipv: int,
    max_moves: int,
    skip_first_ply: int,
    sample_rate: int,
    adjudication_threshold: int = 10,
    adjudication_moves: int = 10,
) -> dict:
    """
    Worker function to run a single game in a subprocess.
    
    Each worker has its own UCI engine instance(s).
    
    Args:
        game_args: Dict with 'game_id' and 'opening' keys.
        engine_path: Path to UCI engine.
        white_depth: Search depth for white player.
        black_depth: Search depth for black player.
        multipv: Number of moves for distribution.
        max_moves: Maximum moves per game.
        skip_first_ply: Skip first N ply positions (opening theory).
        sample_rate: Store every Nth position.
        adjudication_threshold: Centipawns for draw adjudication.
        adjudication_moves: Consecutive moves within threshold for adjudication.
    
    Returns:
        Game data dict with positions and result.
    """
    opening = game_args.get('opening')
    swap_colors = game_args.get('swap_colors', False)
    
    # Setup board from opening
    if opening is not None:
        board = chess.Board(opening.get("fen", chess.STARTING_FEN))
        opening_name = opening.get("name", "Unknown")
    else:
        board = chess.Board()
        opening_name = "Starting Position"
    
    opening_fen = board.fen()
    positions = []
    ply = 0
    eval_history = []
    adjudicated = False
    
    # Determine effective depths (swap if requested for variety)
    if swap_colors:
        eff_white_depth, eff_black_depth = black_depth, white_depth
    else:
        eff_white_depth, eff_black_depth = white_depth, black_depth
    
    # Use analyzer at higher depth for move distribution
    analyzer_depth = max(eff_white_depth, eff_black_depth)
    
    # Create engine(s) for this worker
    # If depths are same, use single engine; otherwise use separate engines
    if eff_white_depth == eff_black_depth:
        with UCIAgent(engine_path, depth=eff_white_depth, multipv=multipv) as engine:
            white_engine = black_engine = analyzer = engine
            result = _play_game(
                board, white_engine, black_engine, analyzer,
                max_moves, skip_first_ply, sample_rate,
                adjudication_threshold, adjudication_moves,
                multipv, positions, eval_history
            )
            adjudicated = result['adjudicated']
            ply = result['ply']
    else:
        with (UCIAgent(engine_path, depth=eff_white_depth, multipv=multipv) as white_engine,
              UCIAgent(engine_path, depth=eff_black_depth, multipv=multipv) as black_engine,
              UCIAgent(engine_path, depth=analyzer_depth, multipv=multipv) as analyzer):
            result = _play_game(
                board, white_engine, black_engine, analyzer,
                max_moves, skip_first_ply, sample_rate,
                adjudication_threshold, adjudication_moves,
                multipv, positions, eval_history
            )
            adjudicated = result['adjudicated']
            ply = result['ply']
    
    # Get result
    game_result = board.result()
    
    # Convert incomplete games to draws
    if game_result == '*':
        game_result = '1/2-1/2'
    
    return {
        'opening_fen': opening_fen,
        'opening_name': opening_name,
        'result': game_result,
        'white_agent': f"Stockfish_d{eff_white_depth}",
        'black_agent': f"Stockfish_d{eff_black_depth}",
        'num_moves': ply,
        'positions': positions,
    }


def _play_game(
    board: chess.Board,
    white_engine: UCIAgent,
    black_engine: UCIAgent,
    analyzer: UCIAgent,
    max_moves: int,
    skip_first_ply: int,
    sample_rate: int,
    adjudication_threshold: int,
    adjudication_moves: int,
    multipv: int,
    positions: list,
    eval_history: list,
) -> dict:
    """Helper to play out the game loop."""
    ply = 0
    adjudicated = False
    
    while not board.is_game_over() and ply < max_moves:
        current_engine = white_engine if board.turn else black_engine
        
        # Get move distribution for training
        move_dist = analyzer.get_move_distribution(board, num_moves=multipv)
        
        # Track evaluation for adjudication
        if move_dist and len(move_dist) > 0:
            best_score = move_dist[0].get('score', 0)
            eval_history.append(best_score)
            
            # Check for draw adjudication
            if len(eval_history) >= adjudication_moves:
                recent = eval_history[-adjudication_moves:]
                if all(abs(e) <= adjudication_threshold for e in recent):
                    adjudicated = True
                    break
        
        # Apply storage optimization
        should_store = (
            ply >= skip_first_ply and
            (ply - skip_first_ply) % sample_rate == 0
        )
        
        if should_store:
            positions.append({
                'fen': board.fen(),
                'ply': ply,
                'move_distribution': move_dist,
            })
        
        # Get and make move
        move = current_engine.get_move(board)
        board.push(move)
        ply += 1
    
    return {'ply': ply, 'adjudicated': adjudicated}


class ParallelMatchRunner:
    """
    Runs games in parallel using multiprocessing.
    
    Each worker process gets its own UCI engine instance, allowing
    true parallel game generation on multi-core machines.
    
    Example:
        runner = ParallelMatchRunner(
            database=db,
            openings_path="config/openings.json",
            num_workers=4,
        )
        stats = runner.run_games(
            engine_path="/opt/homebrew/bin/stockfish",
            num_games=100,
            depth=15,
        )
    """
    
    def __init__(
        self,
        database: ChessDatabase,
        openings_path: Optional[str | Path] = None,
        max_moves: int = 200,
        multipv: int = 5,
        num_workers: int = 4,
        skip_first_ply: int = 8,
        sample_rate: int = 2,
        adjudication_threshold: int = 10,
        adjudication_moves: int = 10,
    ):
        """
        Initialize parallel match runner.
        
        Args:
            database: ChessDatabase for storing results.
            openings_path: Path to JSON file with opening positions.
            max_moves: Maximum moves per game.
            multipv: Number of moves to analyze per position.
            num_workers: Number of parallel worker processes.
            skip_first_ply: Skip first N ply positions (opening theory).
            sample_rate: Store every Nth position (1=all, 2=every other).
            adjudication_threshold: Centipawns threshold for draw adjudication.
            adjudication_moves: Consecutive moves within threshold for adjudication.
        """
        self.database = database
        self.max_moves = max_moves
        self.multipv = multipv
        self.num_workers = num_workers
        self.skip_first_ply = skip_first_ply
        self.sample_rate = sample_rate
        self.adjudication_threshold = adjudication_threshold
        self.adjudication_moves = adjudication_moves
        
        # Load openings
        self.openings = []
        if openings_path is not None:
            self.openings = self._load_openings(openings_path)
    
    def _load_openings(self, path: str | Path) -> list[dict]:
        """Load openings from JSON file."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: Openings file not found: {path}")
            return [{"name": "Starting Position", "fen": chess.STARTING_FEN}]
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data.get("openings", [])
    
    def _save_game_to_database(self, game_data: dict) -> int:
        """
        Save game data to database.
        
        Returns:
            game_id of the saved game.
        """
        # Add game record
        game_id = self.database.add_game(
            opening_fen=game_data['opening_fen'],
            opening_name=game_data['opening_name'],
            result=game_data['result'],
            white_agent=game_data['white_agent'],
            black_agent=game_data['black_agent'],
            num_moves=game_data['num_moves'],
        )
        
        # Add positions with move distributions
        for pos in game_data['positions']:
            self.database.add_position_with_distribution(
                game_id=game_id,
                fen=pos['fen'],
                ply=pos['ply'],
                move_distribution=pos['move_distribution'],
            )
        
        return game_id
    
    def run_games(
        self,
        engine_path: str | Path,
        num_games: int,
        white_depth: int = 15,
        black_depth: int = 10,
        save_every: int = 10,
        verbose: bool = False,
    ) -> dict:
        """
        Run multiple self-play games in parallel.
        
        Args:
            engine_path: Path to UCI engine binary.
            num_games: Number of games to generate.
            white_depth: Search depth for white (higher = stronger).
            black_depth: Search depth for black (lower = weaker for variety).
            save_every: Print progress every N games.
            verbose: If True, print detailed output.
        
        Returns:
            Statistics about generated data.
        """
        # Prepare game arguments with color swapping for variety
        game_args_list = []
        for i in range(num_games):
            opening = None
            if self.openings:
                opening = self.openings[i % len(self.openings)]
            game_args_list.append({
                'game_id': i,
                'opening': opening,
                'swap_colors': (i % 2 == 1),  # Alternate who gets the depth advantage
            })
        
        # Shuffle to distribute openings across workers
        random.shuffle(game_args_list)
        
        # Create worker function with fixed arguments
        worker_fn = partial(
            _run_single_game_worker,
            engine_path=str(engine_path),
            white_depth=white_depth,
            black_depth=black_depth,
            multipv=self.multipv,
            max_moves=self.max_moves,
            skip_first_ply=self.skip_first_ply,
            sample_rate=self.sample_rate,
            adjudication_threshold=self.adjudication_threshold,
            adjudication_moves=self.adjudication_moves,
        )
        
        total_positions = 0
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
        games_completed = 0
        
        print(f"Starting parallel generation with {self.num_workers} workers...")
        
        # Use multiprocessing pool
        # Note: Using 'spawn' context for better compatibility
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for progress updates
            for game_data in pool.imap_unordered(worker_fn, game_args_list):
                # Save to database (in main process)
                self._save_game_to_database(game_data)
                
                # Update statistics
                total_positions += len(game_data['positions'])
                result = game_data['result']
                results[result] = results.get(result, 0) + 1
                games_completed += 1
                
                if games_completed % save_every == 0 or verbose:
                    print(f"Games: {games_completed}/{num_games}, Positions: {total_positions}")
        
        print(f"Parallel generation complete!")
        
        return {
            'num_games': num_games,
            'total_positions': total_positions,
            'results': results,
        }
    
    def run_games_chunked(
        self,
        engine_path: str | Path,
        num_games: int,
        white_depth: int = 15,
        black_depth: int = 10,
        chunk_size: int = 50,
    ) -> dict:
        """
        Run games in chunks to manage memory better for large datasets.
        
        Args:
            engine_path: Path to UCI engine binary.
            num_games: Total number of games to generate.
            white_depth: Search depth for white.
            black_depth: Search depth for black.
            chunk_size: Number of games per chunk.
        
        Returns:
            Statistics about generated data.
        """
        total_positions = 0
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
        
        num_chunks = (num_games + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_games)
            chunk_games = end - start
            
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: games {start + 1}-{end}")
            
            chunk_stats = self.run_games(
                engine_path=engine_path,
                num_games=chunk_games,
                white_depth=white_depth,
                black_depth=black_depth,
                save_every=chunk_games,  # Just print at end of chunk
            )
            
            total_positions += chunk_stats['total_positions']
            for key, value in chunk_stats['results'].items():
                results[key] = results.get(key, 0) + value
        
        return {
            'num_games': num_games,
            'total_positions': total_positions,
            'results': results,
        }
