# Agents Documentation

## Overview

The agents module (`src/agents/`) provides a unified interface for chess-playing entities. All agents implement the abstract `ChessAgent` base class and can be used interchangeably for:

- Playing games
- Generating training data
- Benchmarking model performance

---

## ChessAgent (Base Class)

Location: `src/agents/base.py`

Abstract base class defining the interface for all chess agents.

### Abstract Methods

| Method | Description |
|--------|-------------|
| `get_move(board, time_limit)` | Returns the best move for the current position |
| `name` | Property returning the agent's name |

### Optional Methods

| Method | Default Behavior |
|--------|------------------|
| `get_move_with_info(board, time_limit)` | Returns move with additional metadata |
| `get_move_distribution(board, num_moves, depth)` | Returns top moves with scores |
| `reset()` | Resets internal state |
| `close()` | Cleans up resources |

### Context Manager Support

All agents support the context manager protocol:

```python
with UCIAgent("/path/to/stockfish") as agent:
    move = agent.get_move(board)
```

---

## RandomAgent

Location: `src/agents/random_agent.py`

Agent that plays uniformly random legal moves. Useful for:

- Baseline comparison
- Generating diverse training data
- Testing

### Constructor

```python
RandomAgent(seed: Optional[int] = None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | Optional[int] | Random seed for reproducibility |

### Example Usage

```python
from src.agents import RandomAgent

agent = RandomAgent(seed=42)
move = agent.get_move(board)
```

---

## UCIAgent

Location: `src/agents/uci_agent.py`

Agent that wraps UCI-compatible chess engines (Stockfish, Leela Chess Zero, etc.).

### Constructor

```python
UCIAgent(
    engine_path: str | Path,
    depth: int = 15,
    time_limit: Optional[float] = None,
    skill_level: Optional[int] = None,
    threads: int = 1,
    hash_mb: int = 128,
    multipv: int = 5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_path` | str/Path | - | Path to UCI engine binary |
| `depth` | int | 15 | Default search depth |
| `time_limit` | float | None | Time limit per move (seconds) |
| `skill_level` | int | None | Stockfish skill level (0-20) |
| `threads` | int | 1 | Number of engine threads |
| `hash_mb` | int | 128 | Hash table size in MB |
| `multipv` | int | 5 | Number of PVs for move distribution |

### Methods

#### get_move

```python
move = agent.get_move(board, time_limit=1.0)
```

Returns the best move from the engine.

#### get_move_with_info

```python
info = agent.get_move_with_info(board)
# Returns: {'move': Move, 'score': 105, 'pv': ['e2e4', 'd7d5', ...], 'depth': 15}
```

Returns move with evaluation score, principal variation, and search depth.

#### get_move_distribution

```python
distribution = agent.get_move_distribution(board, num_moves=5, depth=20)
# Returns: [{'move': Move, 'score': 105}, {'move': Move, 'score': 98}, ...]
```

Uses MultiPV mode to get top moves with centipawn scores.

### Example Usage

```python
from src.agents import UCIAgent

with UCIAgent("/opt/homebrew/bin/stockfish", depth=20) as agent:
    # Get best move
    move = agent.get_move(board)
    
    # Get move with evaluation
    info = agent.get_move_with_info(board)
    print(f"Move: {info['move']}, Score: {info['score']} cp")
    
    # Get top 5 moves
    distribution = agent.get_move_distribution(board, num_moves=5)
    for entry in distribution:
        print(f"{entry['move']}: {entry['score']} cp")
```

---

## LearningAgent

Location: `src/agents/learning_agent.py`

Agent that uses a trained PyTorch model for move selection.

### Constructor

```python
LearningAgent(
    model: ChessModel,
    encoder: StateEncoder,
    device: torch.device,
    temperature: float = 0.0,
    top_k: int = 0,
    agent_name: Optional[str] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | ChessModel | - | Trained PyTorch model |
| `encoder` | StateEncoder | - | Matching state encoder |
| `device` | torch.device | - | Inference device |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `top_k` | int | 0 | Sample from top-k moves (0 = all) |
| `agent_name` | str | None | Custom agent name |

### Move Selection Modes

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Greedy (argmax) selection |
| > 0.0 | Softmax sampling |
| > 0.0 + top_k | Top-k sampling |

### Policy Masking

The agent automatically masks illegal moves before selection, ensuring only legal moves can be chosen.

### Methods

#### get_move

```python
move = agent.get_move(board)
```

Returns the selected move based on the model's policy output.

#### get_move_with_info

```python
info = agent.get_move_with_info(board)
# Returns: {'move': Move, 'probability': 0.85, 'value': 0.12}
```

Returns move with its probability and optional value estimate.

#### get_move_distribution

```python
distribution = agent.get_move_distribution(board, num_moves=5)
# Returns: [{'move': Move, 'score': 850, 'probability': 0.85}, ...]
```

Returns top moves with their probabilities (converted to pseudo-centipawn scores).

### Example Usage

```python
from src.agents import LearningAgent
from src.models.factory import create_model, get_encoder_for_model
from src.config import load_config
from src.device import get_device

# Load configuration and create model
config = load_config("config/default.yaml")
model = create_model(config.model)
model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])

# Get encoder
encoder_factory = get_encoder_for_model(config.model.backbone)
encoder = encoder_factory()

# Create agent
device = get_device()
agent = LearningAgent(
    model=model,
    encoder=encoder,
    device=device,
    temperature=0.1,  # Slight exploration
)

# Play a move
move = agent.get_move(board)
```

---

## Agent Comparison

| Agent | Speed | Strength | Use Case |
|-------|-------|----------|----------|
| RandomAgent | Very Fast | Very Weak | Baseline, diverse data |
| UCIAgent | Slow | Configurable | Data generation, benchmarking |
| LearningAgent | Fast | Model-dependent | Inference, self-play |
