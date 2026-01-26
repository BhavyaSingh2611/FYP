# Chess ML Framework

A modular framework for training and benchmarking CNN, Transformer, and GNN architectures for chess move prediction.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Data
```bash
python scripts/generate_data.py --config config/default.yaml --num-games 100
```

### 2. Train a Model
```bash
python scripts/train.py --config config/default.yaml --epochs 10
```

## Project Structure

```
src/
├── chess_env/      # Board wrapper and state encoders
├── models/         # CNN, Transformer, GNN architectures
├── agents/         # Chess agents (Learning, UCI, Random)
├── data/           # Data generation and storage
└── training/       # Training pipeline
```

## Configuration

Edit `config/default.yaml` to change model type, hyperparameters, and paths.

## Supported Models

- **CNN**: ConvNet, ResNet
- **Transformer**: SquareTransformer, PieceTransformer
- **GNN**: GCN, GAT (with static/dynamic/hybrid edges)