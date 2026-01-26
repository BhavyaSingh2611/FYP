"""
Match runner for generating training data through bot-vs-bot games.
"""
import json
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import chess

from .database import ChessDatabase
from ..agents.uci_agent import UCIAgent


class MatchRunner:
    """
    Runs games between chess agents to generate training data.
    
    Features:
        - Load openings from JSON file
        - Run games between agents
        - Collect move distributions at each position
        - Save to SQLite database
        - Position sampling and opening skip for storage optimization
    """
    
    def __init__(
        self,
        database: ChessDatabase,
        openings_path: Optional[str | Path] = None,
        max_moves: int = 200,
        multipv: int = 5,
        skip_first_ply: int = 8,
        sample_rate: int = 2,
        adjudication_threshold: int = 10,
        adjudication_moves: int = 10,
    ):
        """
        Initialize match runner.
        
        Args:
            database: ChessDatabase for storing results.
            openings_path: Path to JSON file with opening positions.
            max_moves: Maximum moves per game.
            multipv: Number of moves to analyze per position.
            skip_first_ply: Skip first N ply positions (opening theory).
            sample_rate: Store every Nth position (1=all, 2=every other).
            adjudication_threshold: Centipawns threshold for draw adjudication.
            adjudication_moves: Consecutive moves within threshold to adjudicate.
        """
        self.database = database
        self.max_moves = max_moves
        self.multipv = multipv
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
    
    def _check_adjudication(
        self, 
        eval_history: list[int], 
    ) -> bool:
        """
        Check if game should be adjudicated as a draw.
        
        Args:
            eval_history: List of centipawn evaluations.
            
        Returns:
            True if game should be adjudicated as draw.
        """
        if len(eval_history) < self.adjudication_moves:
            return False
        
        # Check last N evaluations are all within threshold
        recent = eval_history[-self.adjudication_moves:]
        return all(abs(e) <= self.adjudication_threshold for e in recent)
    
    def run_single_game(
        self,
        white_agent: UCIAgent,
        black_agent: UCIAgent,
        opening: Optional[dict] = None,
        analyze_with: Optional[UCIAgent] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run a single game and collect training data.
        
        Args:
            white_agent: Agent playing white.
            black_agent: Agent playing black.
            opening: Opening position dict with 'fen' and 'name'.
            analyze_with: Engine to use for position analysis (None = use playing agent).
            verbose: If True, print move-by-move output.
        
        Returns:
            Dict with game info and positions.
        """
        # Setup board
        if opening is not None:
            board = chess.Board(opening.get("fen", chess.STARTING_FEN))
            opening_name = opening.get("name", "Unknown")
        else:
            board = chess.Board()
            opening_name = "Starting Position"
        
        opening_fen = board.fen()
        positions = []
        ply = 0
        eval_history = []  # Track evaluations for adjudication
        adjudicated = False
        
        # Get analyzer engine
        analyzer = analyze_with or (white_agent if board.turn else black_agent)
        
        while not board.is_game_over() and ply < self.max_moves:
            # Get current agent
            current_agent = white_agent if board.turn else black_agent
            
            # Get move distribution for training
            move_dist = analyzer.get_move_distribution(
                board, 
                num_moves=self.multipv,
            )
            
            # Track evaluation for adjudication
            if move_dist and len(move_dist) > 0:
                best_score = move_dist[0].get('score', 0)
                eval_history.append(best_score)
                
                # Check for draw adjudication
                if self._check_adjudication(eval_history):
                    adjudicated = True
                    if verbose:
                        print(f"\nGame adjudicated as draw after {ply} moves")
                    break
            
            # Apply storage optimization: skip opening ply and sample positions
            # This reduces storage by ~50% with minimal impact on training quality
            should_store = (
                ply >= self.skip_first_ply and  # Skip opening theory
                (ply - self.skip_first_ply) % self.sample_rate == 0  # Sample every Nth
            )
            
            if should_store:
                positions.append({
                    'fen': board.fen(),
                    'ply': ply,
                    'move_distribution': move_dist,
                })
            
            # Get and make move
            move = current_agent.get_move(board)
            
            if verbose:
                print(f"{ply}. {move.uci()} ", end="" if ply % 2 == 0 else "\n")
            
            board.push(move)
            ply += 1
        
        # Get result
        result = board.result()
        
        # Convert incomplete games to draws (adjudicated or move limit)
        if result == '*':
            result = '1/2-1/2'
        
        if verbose:
            print(f"\nResult: {result}" + (" (adjudicated)" if adjudicated else ""))
        
        return {
            'opening_fen': opening_fen,
            'opening_name': opening_name,
            'result': result,
            'white_agent': white_agent.name,
            'black_agent': black_agent.name,
            'num_moves': ply,
            'positions': positions,
        }
    
    def save_game_to_database(self, game_data: dict) -> int:
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
        depth: int = 15,
        save_every: int = 10,
        verbose: bool = False,
    ) -> dict:
        """
        Run multiple self-play games with a single engine.
        
        Args:
            engine_path: Path to UCI engine binary.
            num_games: Number of games to generate.
            depth: Search depth for analysis.
            save_every: Print progress every N games.
            verbose: If True, print detailed output.
        
        Returns:
            Statistics about generated data.
        """
        total_positions = 0
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
        
        with UCIAgent(engine_path, depth=depth, multipv=self.multipv) as engine:
            for game_num in tqdm(range(num_games), desc="Generating games"):
                # Select random opening
                if self.openings:
                    opening = random.choice(self.openings)
                else:
                    opening = None
                
                # Run game (engine plays against itself)
                game_data = self.run_single_game(
                    white_agent=engine,
                    black_agent=engine,
                    opening=opening,
                    analyze_with=engine,
                    verbose=verbose,
                )
                
                # Save to database
                self.save_game_to_database(game_data)
                
                # Update statistics
                total_positions += len(game_data['positions'])
                results[game_data['result']] = results.get(game_data['result'], 0) + 1
                
                if (game_num + 1) % save_every == 0:
                    print(f"Games: {game_num + 1}/{num_games}, Positions: {total_positions}")
        
        return {
            'num_games': num_games,
            'total_positions': total_positions,
            'results': results,
        }
    
    def run_matches(
        self,
        white_engine_path: str | Path,
        black_engine_path: str | Path,
        num_games: int,
        depth: int = 15,
        analyzer_depth: int = 20,
    ) -> dict:
        """
        Run matches between two different engines.
        
        Args:
            white_engine_path: Path to engine playing white.
            black_engine_path: Path to engine playing black.
            num_games: Number of games.
            depth: Playing depth.
            analyzer_depth: Analysis depth for move distribution.
        
        Returns:
            Statistics about generated data.
        """
        total_positions = 0
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
        
        with (UCIAgent(white_engine_path, depth=depth, multipv=self.multipv) as white_engine,
              UCIAgent(black_engine_path, depth=depth, multipv=self.multipv) as black_engine,
              UCIAgent(white_engine_path, depth=analyzer_depth, multipv=self.multipv) as analyzer):
            
            for game_num in tqdm(range(num_games), desc="Generating games"):
                # Alternate opening selection
                if self.openings:
                    opening = self.openings[game_num % len(self.openings)]
                else:
                    opening = None
                
                game_data = self.run_single_game(
                    white_agent=white_engine,
                    black_agent=black_engine,
                    opening=opening,
                    analyze_with=analyzer,
                )
                
                self.save_game_to_database(game_data)
                
                total_positions += len(game_data['positions'])
                results[game_data['result']] = results.get(game_data['result'], 0) + 1
        
        return {
            'num_games': num_games,
            'total_positions': total_positions,
            'results': results,
        }
