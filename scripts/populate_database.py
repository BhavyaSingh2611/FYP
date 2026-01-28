#!/usr/bin/env python3
"""
Populate chess_dataset_new.db with engine evaluations from parquet file,
and update openings.json with all openings from CSV.

Optimized for large datasets with batch transactions.
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Database schema (same as ChessDatabase)
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
CREATE INDEX IF NOT EXISTS idx_positions_fen ON positions(fen);
"""


def convert_epd_to_fen(epd: str) -> str:
    """
    Convert EPD to full FEN by adding move counters.
    """
    parts = epd.strip().split()
    if len(parts) >= 4:
        return f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} 0 1"
    return epd + " 0 1"


def import_openings_to_json(csv_path: Path, json_path: Path) -> int:
    """
    Import openings from CSV to JSON format.
    """
    openings = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            opening = {
                "name": row["name"],
                "fen": convert_epd_to_fen(row["epd"]),
                "eco": row["eco"],
                "eco_volume": row["eco-volume"],
            }
            openings.append(opening)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"openings": openings}, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(openings)} openings to {json_path}")
    return len(openings)


def get_best_move_from_line(line: str) -> str:
    """Extract the first move from a UCI line."""
    if not line:
        return ""
    moves = line.strip().split()
    return moves[0] if moves else ""


def convert_mate_to_cp(mate) -> Optional[int]:
    """Convert mate score to centipawn equivalent."""
    if mate is None or pd.isna(mate):
        return None
    mate = int(mate)
    base = 10000
    if mate > 0:
        return base - abs(mate) * 10
    else:
        return -(base - abs(mate) * 10)


def import_evaluations_to_db(parquet_path: Path, db_path: Path, batch_size: int = 5000) -> int:
    """
    Import evaluations from parquet file to database.
    
    Uses batch transactions for performance.
    """
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} evaluation rows")
    
    # Group by FEN
    print("Grouping by FEN...")
    grouped = df.groupby('fen')
    total_positions = len(grouped)
    print(f"Found {total_positions:,} unique positions")
    
    # Create database with schema
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    
    # Disable synchronous writes and use WAL for speed
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
    
    positions_imported = 0
    current_game_id = 0
    current_position_id = 0
    
    # Batch data
    games_batch = []
    positions_batch = []
    moves_batch = []
    
    print("Processing positions...")
    
    for i, (fen, group) in enumerate(grouped):
        # Sort by depth (deepest first)
        group_sorted = group.sort_values('depth', ascending=False)
        
        # Get best evaluation
        best_row = group_sorted.iloc[0]
        best_move = get_best_move_from_line(best_row['line'])
        
        if not best_move:
            continue
        
        # Calculate score
        if pd.notna(best_row.get('mate')):
            best_score = convert_mate_to_cp(best_row['mate'])
        elif pd.notna(best_row.get('cp')):
            best_score = int(best_row['cp'])
        else:
            best_score = 0
        
        current_game_id += 1
        current_position_id += 1
        
        # Add game
        games_batch.append((
            current_game_id,
            fen,
            "Engine Evaluation",
            "*",
            "Stockfish",
            "Stockfish",
            0,
        ))
        
        # Side to move
        side_to_move = 0 if ' w ' in fen else 1
        
        # Add position
        positions_batch.append((
            current_position_id,
            current_game_id,
            fen,
            0,  # ply
            best_move,
            best_score,
            side_to_move,
        ))
        
        # Build move distribution
        seen_moves = set()
        rank = 0
        
        for _, row in group_sorted.iterrows():
            move = get_best_move_from_line(row['line'])
            if not move or move in seen_moves:
                continue
            seen_moves.add(move)
            
            if pd.notna(row.get('mate')):
                score = convert_mate_to_cp(row['mate'])
            elif pd.notna(row.get('cp')):
                score = int(row['cp'])
            else:
                score = 0
            
            moves_batch.append((
                current_position_id,
                move,
                score if score is not None else 0,
                rank,
            ))
            rank += 1
        
        positions_imported += 1
        
        # Commit batch
        if len(games_batch) >= batch_size:
            conn.executemany(
                "INSERT INTO games (id, opening_fen, opening_name, result, white_agent, black_agent, num_moves) VALUES (?, ?, ?, ?, ?, ?, ?)",
                games_batch
            )
            conn.executemany(
                "INSERT INTO positions (id, game_id, fen, ply, best_move_uci, best_move_score, side_to_move) VALUES (?, ?, ?, ?, ?, ?, ?)",
                positions_batch
            )
            conn.executemany(
                "INSERT INTO move_distribution (position_id, move_uci, score, rank) VALUES (?, ?, ?, ?)",
                moves_batch
            )
            conn.commit()
            
            games_batch = []
            positions_batch = []
            moves_batch = []
            
            print(f"Processed {i + 1:,}/{total_positions:,} positions ({100 * (i + 1) / total_positions:.1f}%)")
    
    # Final batch
    if games_batch:
        conn.executemany(
            "INSERT INTO games (id, opening_fen, opening_name, result, white_agent, black_agent, num_moves) VALUES (?, ?, ?, ?, ?, ?, ?)",
            games_batch
        )
        conn.executemany(
            "INSERT INTO positions (id, game_id, fen, ply, best_move_uci, best_move_score, side_to_move) VALUES (?, ?, ?, ?, ?, ?, ?)",
            positions_batch
        )
        conn.executemany(
            "INSERT INTO move_distribution (position_id, move_uci, score, rank) VALUES (?, ?, ?, ?)",
            moves_batch
        )
        conn.commit()
    
    conn.close()
    
    print(f"\nImported {positions_imported:,} positions")
    return positions_imported


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    openings_csv = project_root / "data" / "openings.csv"
    openings_json = project_root / "config" / "openings.json"
    parquet_file = project_root / "data" / "train-00000-of-00017.parquet"
    new_db = project_root / "data" / "chess_dataset_new.db"
    
    print("=" * 60)
    print("Chess Database Population Script")
    print("=" * 60)
    
    # Step 1: Update openings.json
    print("\n[1/2] Updating openings.json...")
    if openings_csv.exists():
        import_openings_to_json(openings_csv, openings_json)
    else:
        print(f"Warning: {openings_csv} not found, skipping")
    
    # Step 2: Import evaluations to database
    print("\n[2/2] Importing evaluations to database...")
    if parquet_file.exists():
        if new_db.exists():
            print(f"Removing existing database: {new_db}")
            new_db.unlink()
        
        import_evaluations_to_db(parquet_file, new_db)
    else:
        print(f"Warning: {parquet_file} not found, skipping")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
