# Data Pipeline Documentation

## Overview

The data module (`src/data/`) handles all aspects of training data:

1. **Database**: SQLite storage for positions and move distributions
2. **Dataset**: PyTorch Dataset for loading training samples
3. **Match Runner**: Bot-vs-bot game generation

---

## ChessDatabase

Location: `src/data/database.py`

SQLite database for storing chess training data.

### Schema

#### Games Table

Stores game-level information.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | INTEGER | Primary key |
| `opening_fen` | TEXT | Starting position FEN |
| `opening_name` | TEXT | Opening name |
| `result` | TEXT | Game result (1-0, 0-1, 1/2-1/2) |
| `white_agent` | TEXT | White player name |
| `black_agent` | TEXT | Black player name |
| `num_moves` | INTEGER | Number of moves played |
| `timestamp` | TEXT | Creation timestamp |

#### Positions Table

Stores individual board positions.

| Column | Type | Description |
|--------|------|-------------|
| `position_id` | INTEGER | Primary key |
| `game_id` | INTEGER | Foreign key to games |
| `fen` | TEXT | Board position FEN |
| `ply` | INTEGER | Half-move number |
| `best_move_uci` | TEXT | Best move in UCI format |
| `best_move_score` | INTEGER | Centipawn score |

#### Move Distributions Table

Stores top moves with scores for each position.

| Column | Type | Description |
|--------|------|-------------|
| `distribution_id` | INTEGER | Primary key |
| `position_id` | INTEGER | Foreign key to positions |
| `move_uci` | TEXT | Move in UCI format |
| `score` | INTEGER | Centipawn score |
| `rank` | INTEGER | Move ranking (1 = best) |

### Constructor

```python
ChessDatabase(db_path: str | Path)
```

Creates or opens a database.

### Context Manager Support

```python
with ChessDatabase("data/chess.db") as db:
    db.add_game(...)
```

### Methods

#### Adding Data

```python
# Add a game
game_id = db.add_game(
    opening_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    opening_name="Starting Position",
    result="1-0",
    white_agent="Stockfish",
    black_agent="Stockfish",
    num_moves=45,
)

# Add a position
position_id = db.add_position(
    game_id=game_id,
    fen="r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    ply=3,
    best_move_uci="d2d4",
    best_move_score=25,
)

# Add move distribution
db.add_move_distribution(
    position_id=position_id,
    moves=[
        {"move": "d2d4", "score": 25, "rank": 1},
        {"move": "g1f3", "score": 20, "rank": 2},
        {"move": "f1c4", "score": 15, "rank": 3},
    ]
)

# Add position with distribution in one call
position_id = db.add_position_with_distribution(
    game_id=game_id,
    fen=fen,
    ply=ply,
    move_distribution=moves,
)
```

#### Querying Data

```python
# Get all positions
positions = db.get_all_positions()

# Get position count
count = db.get_position_count()

# Get game count
games = db.get_game_count()

# Get move distribution for a position
distribution = db.get_move_distribution(position_id)

# Get positions with full distributions
positions = db.get_positions_with_distributions(limit=1000, offset=0)
```

---

## ChessDataset

Location: `src/data/dataset.py`

PyTorch Dataset for loading chess training data.

### Features

- Loads positions from SQLite database
- Encodes boards on-the-fly using provided encoder
- Supports soft labels from move distributions
- Optional value targets from game outcomes

### Constructor

```python
ChessDataset(
    db_path: str | Path,
    encoder: StateEncoder,
    use_soft_labels: bool = True,
    include_value: bool = True,
    cache_positions: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | str/Path | - | Path to SQLite database |
| `encoder` | StateEncoder | - | State encoder instance |
| `use_soft_labels` | bool | True | Use soft policy targets from move distribution |
| `include_value` | bool | True | Include value targets |
| `cache_positions` | bool | True | Load all positions into memory |

### Output Format

Each sample is a dictionary:

```python
{
    'input': tensor or dict,       # Encoded board state
    'policy_target': tensor,       # Move target (4672,)
    'value_target': tensor,        # Game outcome (1,) - optional
}
```

### Soft Labels

When `use_soft_labels=True`, policy targets are soft distributions:

```python
# Instead of one-hot [0, 0, 1, 0, 0]
# Soft labels: [0.1, 0.05, 0.7, 0.1, 0.05]
```

Soft labels are created from engine analysis:
1. Centipawn scores are scaled by 100
2. Softmax is applied to create probabilities
3. Multiple moves share probability mass

### Example Usage

```python
from src.data.dataset import ChessDataset
from src.chess_env.encoders import CNNEncoder

encoder = CNNEncoder()
dataset = ChessDataset(
    db_path="data/chess_dataset.db",
    encoder=encoder,
    use_soft_labels=True,
)

# Get a sample
sample = dataset[0]
print(sample['input'].shape)          # (18, 8, 8)
print(sample['policy_target'].shape)  # (4672,)
print(sample['value_target'].shape)   # (1,)
```

---

## DataLoader Creation

### collate_fn

Custom collate function that handles both tensor and dict-based inputs.

### create_dataloader

Factory function for creating a DataLoader.

```python
from src.data.dataset import create_dataloader

loader = create_dataloader(
    db_path="data/chess_dataset.db",
    encoder=encoder,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    use_soft_labels=True,
    include_value=True,
)

for batch in loader:
    inputs = batch['input']
    policy_targets = batch['policy_target']
    value_targets = batch['value_target']
```

---

## MatchRunner

Location: `src/data/match_runner.py`

Runs games between chess agents to generate training data.

### Features

- Loads openings from JSON file
- Runs games between UCI engines
- Collects move distributions at each position
- Saves data to SQLite database

### Constructor

```python
MatchRunner(
    database: ChessDatabase,
    openings_path: Optional[str | Path] = None,
    max_moves: int = 200,
    multipv: int = 5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `database` | ChessDatabase | - | Database for storing results |
| `openings_path` | str/Path | None | Path to openings JSON |
| `max_moves` | int | 200 | Maximum moves per game |
| `multipv` | int | 5 | Number of moves to analyze |

### Methods

#### run_single_game

Runs a single game and returns game data.

```python
game_data = runner.run_single_game(
    white_agent=white,
    black_agent=black,
    opening={"name": "Italian", "fen": "..."},
    analyze_with=analyzer,
    verbose=True,
)
```

Returns:
```python
{
    'opening_fen': str,
    'opening_name': str,
    'result': str,           # "1-0", "0-1", "1/2-1/2"
    'white_agent': str,
    'black_agent': str,
    'num_moves': int,
    'positions': [
        {
            'fen': str,
            'ply': int,
            'move_distribution': [{...}, ...],
        },
        ...
    ]
}
```

#### save_game_to_database

Saves game data to the database.

```python
game_id = runner.save_game_to_database(game_data)
```

#### run_games

Runs multiple self-play games.

```python
stats = runner.run_games(
    engine_path="/opt/homebrew/bin/stockfish",
    num_games=100,
    depth=15,
    save_every=10,
    verbose=False,
)

print(stats)
# {'num_games': 100, 'total_positions': 4500, 'results': {'1-0': 40, '0-1': 35, '1/2-1/2': 25}}
```

#### run_matches

Runs matches between two different engines.

```python
stats = runner.run_matches(
    white_engine_path="/path/to/engine1",
    black_engine_path="/path/to/engine2",
    num_games=50,
    depth=15,
    analyzer_depth=20,
)
```

### Example Usage

```python
from src.data import ChessDatabase, MatchRunner

# Create database
db = ChessDatabase("data/training.db")

# Create runner
runner = MatchRunner(
    database=db,
    openings_path="config/openings.json",
    max_moves=200,
    multipv=5,
)

# Generate games
stats = runner.run_games(
    engine_path="/opt/homebrew/bin/stockfish",
    num_games=1000,
    depth=15,
)

print(f"Generated {stats['total_positions']} training positions")

db.close()
```

---

## Data Generation Workflow

### Step 1: Configure Generation

Edit `config/default.yaml`:

```yaml
data_generation:
  num_games: 1000
  max_moves_per_game: 200
  save_every: 100
```

### Step 2: Run Generation Script

```bash
# Sequential generation
python scripts/generate_data.py --config config/default.yaml --num-games 1000

# Parallel generation (4x faster)
python scripts/generate_data.py --num-games 1000 --parallel --workers 4
```

### Step 3: Verify Data

```python
from src.data import ChessDatabase

db = ChessDatabase("data/chess_dataset.db")
print(f"Games: {db.get_game_count()}")
print(f"Positions: {db.get_position_count()}")
db.close()
```

---

## Parallel Game Generation

Location: `src/data/parallel_runner.py`

For faster data generation on multi-core machines.

### ParallelMatchRunner

```python
from src.data import ChessDatabase, ParallelMatchRunner

db = ChessDatabase("data/training.db")

runner = ParallelMatchRunner(
    database=db,
    openings_path="config/openings.json",
    max_moves=200,
    multipv=5,
    num_workers=4,  # Number of parallel processes
)

stats = runner.run_games(
    engine_path="/opt/homebrew/bin/stockfish",
    num_games=1000,
    depth=15,
)

print(f"Generated {stats['total_positions']} positions")
db.close()
```

### Command Line Usage

```bash
# Enable parallel mode with 4 workers
python scripts/generate_data.py --num-games 1000 --parallel --workers 4

# Use 8 workers for maximum throughput
python scripts/generate_data.py --num-games 5000 --parallel --workers 8 --depth 12
```

### Performance Comparison

| Mode | Workers | Games/Hour (depth 15) |
|------|---------|----------------------|
| Sequential | 1 | ~20 |
| Parallel | 4 | ~70 |
| Parallel | 8 | ~120 |

Note: Actual speeds depend on hardware and engine configuration.

---

## PGN Import

Location: `scripts/import_pgn.py`

Import positions from existing PGN game databases.

### Features

- **Streaming parser**: Handles large PGN files efficiently
- **Position sampling**: Extract every Nth position
- **Skip positions**: Avoid opening theory and trivial endgames
- **Stockfish analysis**: Adds move distributions to all positions

### Command Line Usage

```bash
# Basic import
python scripts/import_pgn.py --input games.pgn --output data/imported.db

# With sampling (every 2nd position, faster)
python scripts/import_pgn.py --input games.pgn --sample-rate 2

# High-quality analysis
python scripts/import_pgn.py --input games.pgn --depth 20 --multipv 10

# Limit number of games
python scripts/import_pgn.py --input lichess_games.pgn --max-games 1000
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | Required | Path to PGN file |
| `--output` | `data/imported.db` | Output database path |
| `--depth` | 15 | Engine analysis depth |
| `--multipv` | 5 | Moves per position |
| `--sample-rate` | 1 | Extract every Nth position |
| `--skip-first` | 4 | Skip first N positions |
| `--skip-last` | 4 | Skip last N positions |
| `--max-games` | None | Maximum games to import |

### Example: Import Lichess Games

```bash
# Download Lichess database
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
zstd -d lichess_db_standard_rated_2024-01.pgn.zst

# Import first 10,000 games
python scripts/import_pgn.py \
    --input lichess_db_standard_rated_2024-01.pgn \
    --output data/lichess_2024_01.db \
    --max-games 10000 \
    --sample-rate 2 \
    --depth 15
```

---

## Value Targets

The dataset supports two modes for value targets:

### Game Outcome Mode (Recommended)

Uses actual game results for value training:

- `1-0` (White wins): +1 for white positions, -1 for black
- `0-1` (Black wins): -1 for white positions, +1 for black
- `1/2-1/2` (Draw): 0 for all positions

```python
dataset = ChessDataset(
    db_path="data/training.db",
    encoder=encoder,
    use_game_outcome=True,  # Use actual game results
)
```

### Centipawn Score Mode

Uses engine evaluation as proxy for value:

```python
dataset = ChessDataset(
    db_path="data/training.db",
    encoder=encoder,
    use_game_outcome=False,  # Use centipawn scores
)
```

### Blended Mode

Combine game outcomes with centipawn scores:

```python
dataset = ChessDataset(
    db_path="data/training.db",
    encoder=encoder,
    use_game_outcome=True,
    blend_value=0.3,  # 70% game outcome + 30% centipawn
)
```

This can help smooth value targets, especially in positions where the game outcome doesn't reflect the position's actual evaluation.

---

## Database Size Estimates

| Games | Positions (approx) | Database Size |
|-------|-------------------|---------------|
| 100 | 4,500 | ~5 MB |
| 1,000 | 45,000 | ~50 MB |
| 10,000 | 450,000 | ~500 MB |
| 100,000 | 4,500,000 | ~5 GB |

Note: Actual sizes depend on move distribution size and game length.

