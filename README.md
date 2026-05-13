# Searchless Chess

A framework for training and evaluating chess-playing neural networks across multiple architectures: CNNs, Transformers, and Graph Neural Networks.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Model Architectures

| Architecture | Backbone | Encoder | Description |
|---|---|---|---|
| ConvNet | `convnet` | CNN (18-plane board tensor) | Simple convolutional network |
| ResNet | `resnet` | CNN (18-plane board tensor) | Residual network with skip connections |
| Square Transformer | `square_transformer` | Tokenized (64 square tokens) | Attention over board squares |
| Piece Transformer | `piece_transformer` | Tokenized (up to 32 piece tokens) | Attention over active pieces |
| GCN | `gcn` | Graph (nodes + edges) | Graph Convolutional Network |
| GAT | `gat` | Graph (nodes + edges) | Graph Attention Network |

All models support three head types: `policy` (move prediction), `value` (position evaluation), or `dual` (both).

## Script Execution

All scripts are designed to be run from the **root directory** of the project, not from within the `scripts/` folder. This ensures the correct resolution of modules and relative paths for data and checkpoints.

```bash
# Good
./scripts/benchmark/resnet_benchmark.sh
python scripts/train.py --model resnet --name baseline

# Bad (will cause issues)
cd scripts/benchmark
./resnet_benchmark.sh
```

## Training

Train a model on labelled positions from Parquet files (Supervised Training):

```bash
# We provide shell scripts to run training schedules for each model:
./scripts/training/resnet_train.sh

# Or you can use the train script manually:
python scripts/train.py --model resnet --epochs 20 --name baseline
python scripts/train.py --model resnet --name baseline  # Train continuously
```

### Device Selection

The framework auto-detects the best available device (CUDA > MPS > CPU).

## Benchmarking

We recommend using the provided benchmark run scripts for the respective model to maintain the same setup as initially tested. There are scripts for both Stockfish benchmarking and Puzzle benchmarking:

### Stockfish Benchmarking

```bash
# Benchmark a model against Stockfish using a predefined test script
./scripts/benchmark/resnet_benchmark.sh

# Or run manually with a specific checkpoint
python scripts/benchmark.py --model resnet --checkpoint runs/models/resnet/resnet_100M_e15.pt
```

### Puzzle Benchmarking

Evaluate a model's Elo using Lichess Puzzles via Stratified Sampling:

```bash
# Recommended: use the predefined script for a model
./scripts/puzzle_benchmark/resnet_puzzle_benchmark.sh

# Or run manually
python scripts/puzzle_benchmark.py --backbone resnet --weights runs/models/resnet/resnet_100M_e15.pt
```

## Run Organisation

Outputs (checkpoints, benchmarks, PGNs) are typically organised under directories within `runs/`:

Please use `python3 scripts/download_models.sh` to populate the runs/models directory

```
runs/
├── models/                       # Downloaded checkpoints
├── <name>/
│   ├── training/<model>/         # checkpoints, metrics
│   └── benchmark/                # benchmark data, PGNs
└── ...
```

## Project Structure

```
├── config/
│   └── settings.yaml             # default training/model configuration
├── data/
│   ├── puzzles/                  # Lichess puzzle parquet dataset
│   └── README.md
├── scripts/
│   ├── benchmark/                # Stockfish benchmark shell scripts
│   ├── puzzle_benchmark/         # Puzzle benchmark shell scripts
│   ├── training/                 # Training schedule shell scripts
│   ├── benchmark.py              # stockfish benchmark script
│   ├── download_models.sh        # download pre-trained checkpoints
│   ├── puzzle_benchmark.py       # puzzle evaluation script
│   └── train.py                  # unified training CLI
├── src/
│   ├── chess_env/
│   │   ├── move_index.py         # UCI move indexing
│   │   └── encoders/             # CNNEncoder, TransformerEncoder, GNNEncoder
│   ├── models/
│   │   ├── base.py               # ChessModel abstract base class
│   │   ├── heads.py              # PolicyHead, ValueHead, DualHead
│   │   ├── factory.py            # create_model() factory
│   │   ├── cnn/                  # ConvNet, ResNet
│   │   ├── transformer/          # SquareTransformer, PieceTransformer
│   │   └── gnn/                  # GCN, GAT
│   ├── agents/
│   │   ├── base.py               # ChessAgent abstract base class
│   │   ├── learning_agent.py     # neural network agent
│   │   ├── uci_agent.py          # Stockfish/UCI engine wrapper
│   │   └── random_agent.py       # random move baseline
│   ├── training/
│   │   ├── trainer.py            # training loop
│   │   └── losses.py             # policy, value, and dual losses
│   ├── data/
│   │   └── dataset.py            # Parquet-based streaming dataset
│   ├── config.py                 # YAML config loader (dataclass-based)
│   └── device.py                 # device auto-detection (CUDA > MPS > CPU)
├── web/
│   ├── client/                   # Svelte frontend
│   ├── server.py                 # Flask server
│   └── start.sh                  # Web app launch script
├── setup.sh                      # venv creation and install script
├── pyproject.toml
└── requirements.txt
```

## Configuration

Model and training parameters are set in `config/settings.yaml` and can be overridden via CLI flags. Key settings:

- **Model**: backbone architecture, head type, layer counts, embedding dimensions
- **Training**: batch size, learning rate, weight decay, LR scheduler
- **Hardware**: device selection, data loader workers

