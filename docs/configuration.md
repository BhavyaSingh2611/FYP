# Configuration Guide

## Overview

The Chess ML Framework uses YAML configuration files for managing all settings. The main configuration is located at `config/default.yaml`.

---

## Configuration Structure

The configuration is organized into the following sections:

### Hardware Configuration

Controls device selection and data loading parallelism.

```yaml
hardware:
  device: auto    # Options: auto, cpu, mps, cuda
  num_workers: 4  # Number of data loading workers
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | string | `auto` | Device for training. `auto` selects MPS (Apple Silicon) > CUDA > CPU |
| `num_workers` | integer | 4 | Number of parallel data loading workers |

---

### Engine Configuration

Settings for external UCI chess engines (e.g., Stockfish).

```yaml
engines:
  stockfish:
    path: /opt/homebrew/bin/stockfish
    default_depth: 15
    default_multipv: 5
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `/opt/homebrew/bin/stockfish` | Path to the UCI engine binary |
| `default_depth` | integer | 15 | Default search depth for analysis |
| `default_multipv` | integer | 5 | Number of top moves to analyze per position |

---

### Path Configuration

File and directory paths for data and checkpoints.

```yaml
paths:
  openings: config/openings.json
  database: data/chess_dataset.db
  checkpoints: checkpoints/
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `openings` | string | `config/openings.json` | JSON file containing opening positions |
| `database` | string | `data/chess_dataset.db` | SQLite database for training data |
| `checkpoints` | string | `checkpoints/` | Directory for model checkpoints |

---

### Model Configuration

Neural network architecture settings.

```yaml
model:
  backbone: resnet
  head: dual
  
  cnn:
    num_blocks: 10
    channels: 256
  
  transformer:
    embed_dim: 256
    num_heads: 8
    num_layers: 6
    dropout: 0.1
  
  gnn:
    hidden_dim: 256
    num_layers: 6
    edge_type: hybrid
    heads: 4
```

#### Backbone Options

| Value | Description |
|-------|-------------|
| `convnet` | Standard 6-layer ConvNet |
| `resnet` | ResNet with skip connections |
| `square_transformer` | Transformer with 64 fixed tokens (one per square) |
| `piece_transformer` | Transformer with variable-length tokens (one per piece) |
| `gcn` | Graph Convolutional Network |
| `gat` | Graph Attention Network |

#### Head Options

| Value | Description |
|-------|-------------|
| `policy` | Policy head only (move prediction) |
| `value` | Value head only (position evaluation) |
| `dual` | Both policy and value heads (AlphaZero-style) |

#### CNN-Specific Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_blocks` | integer | 10 | Number of residual blocks (ResNet: 6, 10, or 20) |
| `channels` | integer | 256 | Number of convolutional channels |

#### Transformer-Specific Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_dim` | integer | 256 | Embedding dimension |
| `num_heads` | integer | 8 | Number of attention heads |
| `num_layers` | integer | 6 | Number of Transformer layers |
| `dropout` | float | 0.1 | Dropout rate |

#### GNN-Specific Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hidden_dim` | integer | 256 | Hidden dimension |
| `num_layers` | integer | 6 | Number of GNN layers |
| `edge_type` | string | `hybrid` | Edge construction: `static`, `dynamic`, or `hybrid` |
| `heads` | integer | 4 | Number of attention heads (GAT only) |

---

### Training Configuration

Hyperparameters for model training.

```yaml
training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
  lr_scheduler:
    type: cosine
    step_size: 10
    gamma: 0.1
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | integer | 256 | Training batch size |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `weight_decay` | float | 0.0001 | L2 regularization weight |
| `epochs` | integer | 50 | Number of training epochs |
| `policy_loss_weight` | float | 1.0 | Weight for policy loss (dual head) |
| `value_loss_weight` | float | 1.0 | Weight for value loss (dual head) |

#### Learning Rate Scheduler

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | `cosine` | Scheduler type: `step`, `cosine`, or `none` |
| `step_size` | integer | 10 | Epochs between LR steps (step scheduler) |
| `gamma` | float | 0.1 | LR decay factor (step scheduler) |

---

### Data Generation Configuration

Settings for bot-vs-bot data generation.

```yaml
data_generation:
  num_games: 1000
  max_moves_per_game: 200
  temperature: 1.0
  save_every: 100
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_games` | integer | 1000 | Number of games to generate |
| `max_moves_per_game` | integer | 200 | Maximum moves per game (prevents infinite games) |
| `temperature` | float | 1.0 | Sampling temperature for move selection |
| `save_every` | integer | 100 | Commit to database every N games |

---

## Openings Configuration

The `config/openings.json` file contains a list of chess opening positions:

```json
{
  "openings": [
    {
      "name": "Starting Position",
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    },
    {
      "name": "Italian Game",
      "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    }
  ]
}
```

Each opening has:
- `name`: Human-readable opening name
- `fen`: FEN string representing the board position

---

## Configuration Classes

The configuration is parsed into Python dataclasses:

| Class | Purpose |
|-------|---------|
| `HardwareConfig` | Device and worker settings |
| `EngineConfig` | External engine settings |
| `PathsConfig` | File path settings |
| `CNNConfig` | CNN architecture parameters |
| `TransformerConfig` | Transformer architecture parameters |
| `GNNConfig` | GNN architecture parameters |
| `ModelConfig` | Combined model configuration |
| `LRSchedulerConfig` | Learning rate scheduler settings |
| `TrainingConfig` | Training hyperparameters |
| `DataGenerationConfig` | Data generation settings |
| `Config` | Root configuration object |

---

## Utility Functions

### load_config

```python
from src.config import load_config

config = load_config("config/default.yaml")
```

Loads a YAML configuration file and returns a `Config` object.

### load_openings

```python
from src.config import load_openings

openings = load_openings("config/openings.json")
```

Loads opening positions from a JSON file.

### get_config_value

```python
from src.config import get_config_value

backbone = get_config_value(config, "model.backbone")
```

Retrieves a configuration value using a dot-separated path.
