# API Reference

## Overview

This document provides a quick reference for all public APIs in the Chess ML Framework.

---

## Configuration Module

Location: `src/config.py`

### Functions

| Function | Description |
|----------|-------------|
| `load_config(config_path)` | Load configuration from YAML file |
| `load_openings(openings_path)` | Load openings from JSON file |
| `get_config_value(config, key_path)` | Get config value by dot path |

### Dataclasses

| Class | Description |
|-------|-------------|
| `Config` | Root configuration object |
| `HardwareConfig` | Device and worker settings |
| `EngineConfig` | Chess engine settings |
| `PathsConfig` | File path settings |
| `ModelConfig` | Neural network settings |
| `CNNConfig` | CNN-specific settings |
| `TransformerConfig` | Transformer-specific settings |
| `GNNConfig` | GNN-specific settings |
| `TrainingConfig` | Training hyperparameters |
| `LRSchedulerConfig` | Scheduler settings |
| `DataGenerationConfig` | Data generation settings |

---

## Device Module

Location: `src/device.py`

### Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `get_device` | `(force_cpu=False, verbose=True)` | `torch.device` |
| `get_device_from_config` | `(config, verbose=True)` | `torch.device` |
| `optimize_for_device` | `(model, device)` | `nn.Module` |

---

## Agents Module

Location: `src/agents/`

### Base Class

```python
class ChessAgent(ABC):
    def get_move(self, board, time_limit=None) -> chess.Move
    @property
    def name(self) -> str
    def get_move_with_info(self, board, time_limit=None) -> dict
    def get_move_distribution(self, board, num_moves=5, depth=None) -> list[dict]
    def reset(self) -> None
    def close(self) -> None
```

### Implementations

| Class | Constructor |
|-------|-------------|
| `RandomAgent` | `(seed=None)` |
| `UCIAgent` | `(engine_path, depth=15, time_limit=None, skill_level=None, threads=1, hash_mb=128, multipv=5)` |
| `LearningAgent` | `(model, encoder, device, temperature=0.0, top_k=0, agent_name=None)` |

---

## Chess Environment Module

Location: `src/chess_env/`

### BoardWrapper

```python
class BoardWrapper:
    def __init__(self, fen=None)
    @property fen, turn, is_game_over, result, legal_moves, legal_moves_uci
    def push(self, move) -> None
    def pop(self) -> chess.Move
    def copy(self) -> BoardWrapper
    def reset(self) -> None
    def set_fen(self, fen) -> None
    @staticmethod move_to_index(move) -> int
    @staticmethod index_to_move(index) -> str
```

### Constants

| Constant | Type | Description |
|----------|------|-------------|
| `ALL_UCI_MOVES` | list[str] | All possible UCI moves |
| `UCI_MOVE_TO_INDEX` | dict[str, int] | Move to index mapping |
| `INDEX_TO_UCI_MOVE` | dict[int, str] | Index to move mapping |
| `NUM_MOVES` | int | Total move count (4672) |

---

## Encoders Module

Location: `src/chess_env/encoders/`

### Base Class

```python
class StateEncoder(ABC):
    def encode(self, board) -> Any
    def get_input_shape(self) -> tuple
    @property
    def name(self) -> str
```

### Implementations

| Class | Constructor | Output Shape |
|-------|-------------|--------------|
| `CNNEncoder` | `(flip_perspective=True)` | `(18, 8, 8)` tensor |
| `TransformerEncoder` | `(tokenizer_type="square")` | dict |
| `GNNEncoder` | `(edge_type="hybrid")` | dict |

---

## Models Module

Location: `src/models/`

### Base Class

```python
class ChessModel(ABC, nn.Module):
    def forward_backbone(self, x) -> torch.Tensor
    def get_backbone_output_dim(self) -> int
    def forward(self, x) -> dict
    def set_head(self, head) -> None
    @property
    def name(self) -> str
    def count_parameters(self) -> int
```

### CNN Models

| Class | Constructor |
|-------|-------------|
| `ConvNet` | `(input_channels=18, channels=256, num_layers=6)` |
| `ResNet` | `(input_channels=18, channels=256, num_blocks=10)` |

### Transformer Models

| Class | Constructor |
|-------|-------------|
| `SquareTransformer` | `(vocab_size=13, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1, use_cls_token=True)` |
| `PieceTransformer` | `(vocab_size=12, max_pieces=32, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1)` |

### GNN Models

| Class | Constructor |
|-------|-------------|
| `GCN` | `(input_dim=12, hidden_dim=256, num_layers=6, edge_type="hybrid", dropout=0.1)` |
| `GAT` | `(input_dim=12, hidden_dim=256, num_layers=6, edge_type="hybrid", heads=4, dropout=0.1)` |

### Output Heads

| Class | Constructor | Output |
|-------|-------------|--------|
| `PolicyHead` | `(input_dim, hidden_dim=256)` | `{'policy': (B, 4672)}` |
| `ValueHead` | `(input_dim, hidden_dim=256)` | `{'value': (B, 1)}` |
| `DualHead` | `(input_dim, hidden_dim=256)` | `{'policy': ..., 'value': ...}` |

### Factory Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `create_model` | `(config)` | `ChessModel` |
| `create_backbone` | `(backbone_type, **kwargs)` | `ChessModel` |
| `get_encoder_for_model` | `(backbone_type)` | Encoder class |
| `create_head` | `(head_type, input_dim, hidden_dim=256)` | `nn.Module` |
| `list_available_models` | `()` | `dict` |

---

## Data Module

Location: `src/data/`

### ChessDatabase

```python
class ChessDatabase:
    def __init__(self, db_path)
    def add_game(...) -> int
    def add_position(...) -> int
    def add_move_distribution(position_id, moves) -> None
    def add_position_with_distribution(...) -> int
    def get_all_positions() -> list
    def get_position_count() -> int
    def get_game_count() -> int
    def get_move_distribution(position_id) -> list
    def get_positions_with_distributions(limit=None, offset=0) -> list
    def close() -> None
```

### ChessDataset

```python
class ChessDataset(Dataset):
    def __init__(self, db_path, encoder, use_soft_labels=True, include_value=True, cache_positions=True)
    def __len__(self) -> int
    def __getitem__(self, idx) -> dict
```

### MatchRunner

```python
class MatchRunner:
    def __init__(self, database, openings_path=None, max_moves=200, multipv=5)
    def run_single_game(white_agent, black_agent, opening=None, analyze_with=None, verbose=False) -> dict
    def save_game_to_database(game_data) -> int
    def run_games(engine_path, num_games, depth=15, save_every=10, verbose=False) -> dict
    def run_matches(white_engine_path, black_engine_path, num_games, depth=15, analyzer_depth=20) -> dict
```

### Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `collate_fn` | `(batch)` | `dict` |
| `create_dataloader` | `(db_path, encoder, batch_size=256, shuffle=True, num_workers=0, use_soft_labels=True, include_value=True)` | `DataLoader` |

---

## Training Module

Location: `src/training/`

### Loss Classes

| Class | Constructor |
|-------|-------------|
| `PolicyLoss` | `(use_soft_labels=True, label_smoothing=0.0)` |
| `ValueLoss` | `()` |
| `DualLoss` | `(policy_weight=1.0, value_weight=1.0, use_soft_labels=True)` |

### Trainer

```python
class Trainer:
    def __init__(self, model, device, head_type="dual", learning_rate=0.001, weight_decay=0.0001, ...)
    def train_epoch(self, dataloader, epoch, log_interval=100) -> dict
    def evaluate(self, dataloader) -> dict
    def train(self, train_loader, val_loader=None, epochs=50, scheduler_type="cosine", save_best=True, save_every=10) -> dict
    def save_checkpoint(self, filename) -> None
    def load_checkpoint(self, path) -> None
```

### Factory Function

| Function | Signature | Returns |
|----------|-----------|---------|
| `create_loss` | `(head_type, policy_weight=1.0, value_weight=1.0, use_soft_labels=True)` | `nn.Module` |

---

## Import Shortcuts

The package provides convenient import shortcuts:

```python
# Agents
from src.agents import ChessAgent, RandomAgent, UCIAgent, LearningAgent

# Encoders
from src.chess_env.encoders import StateEncoder, CNNEncoder, TransformerEncoder, GNNEncoder

# Models
from src.models import ChessModel
from src.models.factory import create_model, create_backbone, get_encoder_for_model
from src.models.cnn import ConvNet, ResNet
from src.models.transformer import SquareTransformer, PieceTransformer
from src.models.gnn import GCN, GAT

# Data
from src.data import ChessDatabase, ChessDataset, MatchRunner

# Training
from src.training import Trainer, PolicyLoss, ValueLoss, DualLoss
```
