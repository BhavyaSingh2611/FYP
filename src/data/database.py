"""
SQLite database for storing chess game data and training positions.
"""
import sqlite3
from pathlib import Path
from typing import Optional
import json

import chess


class ChessDatabase:
    """
    SQLite database for chess training data.
    
    Schema:
        - games: Game-level information (opening, result, agents)
        - positions: Individual board positions with best move
        - move_distribution: Multiple moves per position with scores
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        opening_fen TEXT,
        opening_name TEXT,
        result TEXT,
        white_agent TEXT,
        black_agent TEXT,
        num_moves INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL,
        fen TEXT NOT NULL,
        ply INTEGER NOT NULL,
        best_move_uci TEXT NOT NULL,
        best_move_score INTEGER,
        side_to_move INTEGER,
        FOREIGN KEY (game_id) REFERENCES games(id)
    );
    
    CREATE TABLE IF NOT EXISTS move_distribution (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER NOT NULL,
        move_uci TEXT NOT NULL,
        score INTEGER NOT NULL,
        rank INTEGER NOT NULL,
        FOREIGN KEY (position_id) REFERENCES positions(id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_positions_game ON positions(game_id);
    CREATE INDEX IF NOT EXISTS idx_move_dist_position ON move_distribution(position_id);
    """
    
    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def add_game(
        self,
        opening_fen: str,
        opening_name: str,
        result: str,
        white_agent: str,
        black_agent: str,
        num_moves: int,
    ) -> int:
        """
        Add a game record.
        
        Returns:
            game_id of the inserted game.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO games (opening_fen, opening_name, result, white_agent, black_agent, num_moves)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (opening_fen, opening_name, result, white_agent, black_agent, num_moves),
        )
        conn.commit()
        return cursor.lastrowid
    
    def add_position(
        self,
        game_id: int,
        fen: str,
        ply: int,
        best_move_uci: str,
        best_move_score: Optional[int] = None,
    ) -> int:
        """
        Add a position record.
        
        Returns:
            position_id of the inserted position.
        """
        # Determine side to move (0 = white, 1 = black)
        side_to_move = 0 if ' w ' in fen else 1
        
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO positions (game_id, fen, ply, best_move_uci, best_move_score, side_to_move)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_id, fen, ply, best_move_uci, best_move_score, side_to_move),
        )
        conn.commit()
        return cursor.lastrowid
    
    def add_move_distribution(
        self,
        position_id: int,
        moves: list[dict],
    ) -> None:
        """
        Add move distribution for a position.
        
        Args:
            position_id: ID of the position.
            moves: List of dicts with 'move', 'score', and optional 'rank'.
        """
        conn = self._get_connection()
        
        for rank, move_info in enumerate(moves):
            move = move_info['move']
            move_uci = move.uci() if isinstance(move, chess.Move) else str(move)
            score = move_info.get('score', 0)
            
            conn.execute(
                """
                INSERT INTO move_distribution (position_id, move_uci, score, rank)
                VALUES (?, ?, ?, ?)
                """,
                (position_id, move_uci, score, rank),
            )
        
        conn.commit()
    
    def add_position_with_distribution(
        self,
        game_id: int,
        fen: str,
        ply: int,
        move_distribution: list[dict],
    ) -> int:
        """
        Add a position with its move distribution in one call.
        
        Args:
            game_id: Game ID.
            fen: Board position FEN.
            ply: Half-move number.
            move_distribution: List of moves with scores.
        
        Returns:
            position_id.
        """
        if not move_distribution:
            return -1
        
        # Best move is the highest-scored move
        best_move = move_distribution[0]
        best_move_uci = best_move['move'].uci() if isinstance(best_move['move'], chess.Move) else str(best_move['move'])
        best_move_score = best_move.get('score', 0)
        
        position_id = self.add_position(
            game_id=game_id,
            fen=fen,
            ply=ply,
            best_move_uci=best_move_uci,
            best_move_score=best_move_score,
        )
        
        self.add_move_distribution(position_id, move_distribution)
        
        return position_id
    
    def get_all_positions(self) -> list[dict]:
        """Get all positions with their best moves."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT p.id, p.fen, p.best_move_uci, p.best_move_score, p.ply, p.side_to_move,
                   g.white_agent, g.black_agent
            FROM positions p
            JOIN games g ON p.game_id = g.id
            """
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_position_count(self) -> int:
        """Get total number of positions."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM positions")
        return cursor.fetchone()[0]
    
    def get_game_count(self) -> int:
        """Get total number of games."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        return cursor.fetchone()[0]
    
    def get_move_distribution(self, position_id: int) -> list[dict]:
        """Get move distribution for a position."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT move_uci, score, rank
            FROM move_distribution
            WHERE position_id = ?
            ORDER BY rank
            """,
            (position_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_positions_with_distributions(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get positions with their full move distributions.
        
        Args:
            limit: Maximum number of positions to return.
            offset: Offset for pagination.
        
        Returns:
            List of positions with move distributions.
        """
        conn = self._get_connection()
        
        query = """
            SELECT p.id, p.fen, p.best_move_uci, p.best_move_score, p.ply, p.side_to_move
            FROM positions p
            ORDER BY p.id
        """
        
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        cursor = conn.execute(query)
        positions = [dict(row) for row in cursor.fetchall()]
        
        # Fetch move distributions for each position
        for pos in positions:
            pos['move_distribution'] = self.get_move_distribution(pos['id'])
        
        return positions
    
    def get_positions_with_outcomes(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get positions with their full move distributions AND game outcomes.
        
        This is the preferred method for training as it includes the actual
        game result for proper value target calculation.
        
        Args:
            limit: Maximum number of positions to return.
            offset: Offset for pagination.
        
        Returns:
            List of positions with move distributions and game results.
        """
        conn = self._get_connection()
        
        query = """
            SELECT p.id, p.fen, p.best_move_uci, p.best_move_score, p.ply, p.side_to_move,
                   g.result as game_result, g.white_agent, g.black_agent
            FROM positions p
            JOIN games g ON p.game_id = g.id
            ORDER BY p.id
        """
        
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        cursor = conn.execute(query)
        positions = [dict(row) for row in cursor.fetchall()]
        
        # Fetch move distributions for each position
        for pos in positions:
            pos['move_distribution'] = self.get_move_distribution(pos['id'])
        
        return positions
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
