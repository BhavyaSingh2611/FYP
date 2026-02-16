# Chess ML Framework

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

## Training

### Supervised Training

Train a model on labelled positions from Parquet files:

```bash
# Train a ResNet for 20 epochs
python scripts/train.py supervised --model resnet --epochs 20 --name baseline

# Train on a specific dataset, limiting to 100k positions
python scripts/train.py supervised --model convnet --epochs 10 \
  --database data/training.parquet --num-samples 100000 --name small_run

# Train on a directory of Parquet files
python scripts/train.py supervised --model square_transformer \
  --database data/ --epochs 30 --name transformer_run
```

### Self-Play (Reinforcement Learning)

Improve a pre-trained model via MCTS self-play:

```bash
# Run 5 iterations of self-play with 20 games each
python scripts/train.py self-play --model resnet --games 20 --iterations 5 --name rl_run

# Quick test to verify self-play works
python scripts/train.py self-play --model convnet --dry-run --name test
```

### Device Selection

The framework auto-detects the best available device (CUDA > MPS > CPU). Override with:

```bash
python scripts/train.py --device cpu supervised --model resnet --epochs 5
```

## Benchmarking

Benchmark models against Stockfish with move-by-move evaluation:

```bash
# Benchmark all models from a named run
python scripts/benchmark.py --name baseline --games 4

# Benchmark a single model
python scripts/benchmark.py --model resnet --name baseline --games 6

# Benchmark with a specific checkpoint
python scripts/benchmark.py --model convnet --checkpoint path/to/model.pt --games 4

# Adjust Stockfish difficulty
python scripts/benchmark.py --name baseline --opponent-depth 3 --skill-level 5
```

Benchmark outputs include:
- PGN files (viewable in Lichess, Chess.com, or any chess software)
- Evaluation flow charts per game
- Comparison plots across models
- Markdown report with results summary

## Run Organisation

Using `--name` organises all outputs under a single directory:

```
runs/<name>/
├── training/
│   ├── convnet/          # checkpoints, metrics
│   ├── resnet/
│   └── ...
├── self_play/            # self-play checkpoints and logs
└── benchmark/
    ├── pgn/              # individual PGN files
    ├── figures/          # evaluation charts
    ├── all_games.pgn     # combined PGN
    └── detailed_results.json
```

## Project Structure

```
├── config/
│   └── default.yaml              # default training/model configuration
├── scripts/
│   ├── train.py                  # unified training CLI (supervised + self-play)
│   └── benchmark.py              # benchmark models against Stockfish
├── src/
│   ├── chess_env/
│   │   ├── board_wrapper.py      # board state wrapper, UCI move indexing
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
│   │   ├── mcts_agent.py         # MCTS with neural network guidance
│   │   ├── uci_agent.py          # Stockfish/UCI engine wrapper
│   │   └── random_agent.py       # random move baseline
│   ├── training/
│   │   ├── trainer.py            # training loop
│   │   ├── losses.py             # policy, value, and dual losses
│   │   └── self_play.py          # MCTS self-play data generation
│   ├── data/
│   │   └── dataset.py            # Parquet-based streaming dataset
│   ├── config.py                 # YAML config loader (dataclass-based)
│   └── device.py                 # device auto-detection (CUDA > MPS > CPU)
├── setup.sh                      # venv creation and install script
├── pyproject.toml
└── requirements.txt
```

## Configuration

Model and training parameters are set in `config/default.yaml` and can be overridden via CLI flags. Key settings:

- **Model**: backbone architecture, head type, layer counts, embedding dimensions
- **Training**: batch size, learning rate, weight decay, LR scheduler
- **Hardware**: device selection, data loader workers

## Requirements

- Python 3.10+
- PyTorch
- python-chess
- pyarrow (for Parquet data loading)
- torch_geometric (for GNN models)
- Stockfish binary (for benchmarking)
