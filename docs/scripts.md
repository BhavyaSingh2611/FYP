# Scripts Documentation

## Overview

The scripts directory (`scripts/`) contains entry point scripts for common tasks:

1. **generate_data.py**: Generate training data using bot-vs-bot games
2. **train.py**: Train chess models

---

## generate_data.py

Generates training data by running bot-vs-bot games using Stockfish.

### Usage

```bash
python scripts/generate_data.py [options]
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/default.yaml` | Path to configuration file |
| `--num-games` | int | Config value | Number of games to generate |
| `--depth` | int | Config value | Engine search depth |
| `--output` | str | Config value | Output database path |
| `--verbose` | flag | False | Print detailed output |

### Examples

#### Basic Usage

```bash
python scripts/generate_data.py
```

Uses all settings from `config/default.yaml`.

#### Custom Number of Games

```bash
python scripts/generate_data.py --num-games 500
```

#### Full Customization

```bash
python scripts/generate_data.py \
    --config config/my_config.yaml \
    --num-games 1000 \
    --depth 20 \
    --output data/my_dataset.db \
    --verbose
```

### Output

```
Chess Training Data Generator
========================================
Engine: /opt/homebrew/bin/stockfish
Depth: 15
MultiPV: 5
Games: 100
Database: data/chess_dataset.db
Openings: config/openings.json

Starting data generation...
Generating games: 100%|████████████████| 100/100 [05:23<00:00, 3.23s/game]

Generation Complete!
========================================
Games generated: 100
Positions saved: 4523
Results: {'1-0': 42, '0-1': 38, '1/2-1/2': 20}
Database size: 5.2 MB
```

### Process Flow

1. Load configuration from YAML file
2. Create or open SQLite database
3. Initialize MatchRunner with openings
4. For each game:
   - Select random opening position
   - Play game using Stockfish vs Stockfish
   - Analyze each position for move distribution
   - Save positions and distributions to database
5. Print statistics

---

## train.py

Trains chess models using data from the database.

### Usage

```bash
python scripts/train.py [options]
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/default.yaml` | Path to configuration file |
| `--epochs` | int | Config value | Number of training epochs |
| `--batch-size` | int | Config value | Training batch size |
| `--model` | str | Config value | Model backbone type |
| `--head` | str | Config value | Output head type |
| `--database` | str | Config value | Path to training database |
| `--checkpoint` | str | None | Checkpoint to resume from |
| `--output-dir` | str | Config value | Checkpoint output directory |

### Model Options

| Value | Description |
|-------|-------------|
| `convnet` | Standard 6-layer ConvNet |
| `resnet` | ResNet with skip connections |
| `square_transformer` | Square-based Transformer |
| `piece_transformer` | Piece-based Transformer |
| `gcn` | Graph Convolutional Network |
| `gat` | Graph Attention Network |

### Head Options

| Value | Description |
|-------|-------------|
| `policy` | Policy head only |
| `value` | Value head only |
| `dual` | Both policy and value heads |

### Examples

#### Basic Training

```bash
python scripts/train.py
```

Uses all settings from `config/default.yaml`.

#### Train ResNet for 100 Epochs

```bash
python scripts/train.py --model resnet --epochs 100
```

#### Train Transformer with Custom Batch Size

```bash
python scripts/train.py --model square_transformer --batch-size 128
```

#### Resume from Checkpoint

```bash
python scripts/train.py --checkpoint checkpoints/ResNet_10B_256C_epoch_20.pt
```

#### Full Customization

```bash
python scripts/train.py \
    --config config/my_config.yaml \
    --model gat \
    --head dual \
    --epochs 50 \
    --batch-size 256 \
    --database data/my_dataset.db \
    --output-dir checkpoints/gat_experiment/
```

### Output

```
Chess Model Trainer
========================================
Device: MPS (Apple Silicon)
Model: resnet + dual head
Database: data/chess_dataset.db
Epochs: 50
Batch size: 256
Learning rate: 0.001

Creating model...
Model: ResNet_10B_256C
Parameters: 1,234,567

Encoder: CNNEncoder

Loading dataset...
Dataset size: 45000 positions

Starting training...
Epoch 1/50: 100%|████████████████| 176/176 [00:45<00:00, 3.91batch/s]
  Loss: 2.3456, Policy: 1.8765, Value: 0.4691
Epoch 2/50: 100%|████████████████| 176/176 [00:44<00:00, 3.95batch/s]
  Loss: 1.5678, Policy: 1.2345, Value: 0.3333
...
Epoch 50/50: 100%|████████████████| 176/176 [00:43<00:00, 4.02batch/s]
  Loss: 0.4567, Policy: 0.3456, Value: 0.1111
Saved checkpoint: checkpoints/ResNet_10B_256C_epoch_50.pt

Training complete!
```

### Process Flow

1. Load configuration from YAML file
2. Select compute device (MPS/CUDA/CPU)
3. Create model using factory
4. Get matching encoder for model type
5. Create data loader from database
6. Initialize trainer with loss function
7. Optionally load checkpoint
8. Train for specified epochs:
   - Forward pass through model
   - Compute loss (policy and/or value)
   - Backward pass and optimizer step
   - Update learning rate scheduler
   - Log metrics
9. Save checkpoints periodically

---

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# Generate small dataset
python scripts/generate_data.py --num-games 100

# Quick training run
python scripts/train.py --epochs 10 --batch-size 64
```

### Workflow 2: Full Training Pipeline

```bash
# Generate large dataset
python scripts/generate_data.py --num-games 10000 --depth 20

# Train model
python scripts/train.py --epochs 100 --model resnet

# Continue training if needed
python scripts/train.py --checkpoint checkpoints/ResNet_10B_256C_best.pt --epochs 50
```

### Workflow 3: Model Comparison

```bash
# Train ConvNet
python scripts/train.py --model convnet --output-dir checkpoints/convnet/

# Train ResNet
python scripts/train.py --model resnet --output-dir checkpoints/resnet/

# Train Transformer
python scripts/train.py --model square_transformer --output-dir checkpoints/transformer/

# Train GNN
python scripts/train.py --model gat --output-dir checkpoints/gat/
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Select specific CUDA device |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Control MPS memory usage |

### Example

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python scripts/train.py

# Limit MPS memory
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/train.py
```

---

## Troubleshooting

### Issue: Engine Not Found

```
FileNotFoundError: Engine not found at /opt/homebrew/bin/stockfish
```

**Solution**: Update `engines.stockfish.path` in config or install Stockfish:
```bash
brew install stockfish
```

### Issue: Database Not Found

```
FileNotFoundError: Database not found
```

**Solution**: Run data generation first:
```bash
python scripts/generate_data.py --num-games 100
```

### Issue: Out of Memory

```
RuntimeError: MPS backend out of memory
```

**Solution**: Reduce batch size:
```bash
python scripts/train.py --batch-size 64
```

### Issue: Slow Training on CPU

**Solution**: Check device selection:
```python
from src.device import get_device
device = get_device(verbose=True)
```

Install appropriate dependencies for GPU acceleration.
