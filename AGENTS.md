# Chess ML Framework - Agent Instructions

## Commands
- **Setup**: `./setup.sh` (creates venv and installs in editable mode)
- **Train (supervised)**: `python scripts/train.py supervised --model resnet --epochs 10 --name my_run`
- **Train (self-play)**: `python scripts/train.py self-play --model resnet --games 20 --name my_run`
- **Benchmark**: `python scripts/benchmark.py --name my_run --games 4`
- **Run tests**: `pytest` (single test: `pytest path/to/test.py::test_name -v`)

## Architecture
- `src/chess_env/` - Board wrapper (`BoardWrapper`) and state encoders (`CNNEncoder`, `TransformerEncoder`, `GNNEncoder`)
- `src/models/` - Model architectures: CNN (`ConvNet`, `ResNet`), Transformer (`SquareTransformer`, `PieceTransformer`), GNN (`GCN`, `GAT`), plus shared `PolicyHead`/`ValueHead`/`DualHead` and model `factory`
- `src/agents/` - `ChessAgent` base class + `RandomAgent`, `UCIAgent`, `LearningAgent`, `MCTSAgent`
- `src/data/` - `ChessDataset` (Parquet-based `IterableDataset`, streams row-groups) and `create_dataloader`
- `src/training/` - `Trainer`, loss functions (`PolicyLoss`, `ValueLoss`, `DualLoss`), `SelfPlayGenerator`
- `src/config.py` - Dataclass-based YAML config loader
- `src/device.py` - Device auto-detection (CUDA > MPS > CPU)
- `config/` - YAML configuration files (`default.yaml`)
- `scripts/train.py` - Unified training CLI (supervised + self-play subcommands)
- `scripts/benchmark.py` - Benchmark models against Stockfish with evaluation tracking and PGN export

## Run Organisation
All scripts accept `--name <run_name>` which organises outputs under:
```
runs/<name>/
  training/<model>/   # supervised checkpoints
  self_play/          # self-play checkpoints
  benchmark/          # benchmark results, PGNs, figures
```

## Training Target
- Primary training hardware: **NVIDIA GPU** (remote VM). Always default to CUDA-optimised paths.
- `torch.compile` is enabled by default in `Trainer`.
- Mixed precision (float16) is enabled by default.

## Code Style
- Python 3.10+, use type hints (`str | Path`, `int | None`)
- Use dataclasses for configs (see `src/config.py`)
- Imports: stdlib → third-party (torch, chess, yaml) → local (src.*)
- Use `Path` from pathlib for file paths
- Agents implement `ChessAgent` base class with `get_move()` and `name` property
- Models extend `ChessModel` base class (PyTorch `nn.Module`)
- Encoders extend `StateEncoder` base class with `encode()` method
- Config via YAML files loaded with `load_config()` from `src.config`
- No comments unless code is complex; explanations go in responses, not code
