# Chess ML Framework - Agent Instructions

## Commands
- **Install**: `pip install -r requirements.txt` or `pip install -e .`
- **Train model**: `python scripts/train.py --config config/default.yaml --epochs 10`
- **Run tests**: `pytest` (single test: `pytest path/to/test.py::test_name -v`)

## Architecture
- `src/chess_env/` - Board wrapper and state encoders
- `src/models/` - CNN (ConvNet, ResNet), Transformer, GNN (GCN, GAT) architectures
- `src/agents/` - ChessAgent base class + RandomAgent, UCIAgent, LearningAgent
- `src/data/` - Training Data in Paraquet files
- `src/training/` - Training pipeline
- `config/` - YAML configs (default.yaml, gpu_training.yaml)

## Code Style
- Python 3.8+, use type hints (`str | Path`, `Optional[int]`)
- Use dataclasses for configs (see `src/config.py`)
- Imports: stdlib → third-party (torch, chess, yaml) → local (src.*)
- Use `Path` from pathlib for file paths
- Agents implement `ChessAgent` base class with `get_move()` and `name` property
- Models use PyTorch; encoders convert board state to tensors
- Config via YAML files loaded with `load_config()` from `src/config`
