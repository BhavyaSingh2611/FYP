# Chess ML Framework - Project Overview

## Introduction

This project is a modular framework for training and benchmarking deep learning models for chess move prediction. The framework supports three major neural network architecture families:

- **Convolutional Neural Networks (CNN)**: ConvNet and ResNet variants
- **Transformers**: Square-based and Piece-based tokenization schemes
- **Graph Neural Networks (GNN)**: GCN and GAT with multiple edge types

The framework is designed for a Final Year Project titled "Training Chess Bots with Various Machine Learning Techniques."

---

## Key Features

1. **Multiple Model Architectures**: Hot-swappable backbones with a unified interface
2. **Flexible State Encoders**: Architecture-specific board encoding (bitboards, tokens, graphs)
3. **Multiple Output Heads**: Policy-only, Value-only, or Dual-headed (AlphaZero-style)
4. **Data Generation Pipeline**: Bot-vs-bot games using Stockfish for labeled training data
5. **Device Optimization**: Automatic device selection with Apple Silicon MPS prioritization
6. **Configuration-Driven**: YAML-based configuration for all hyperparameters

---

## Project Structure

```
project/
├── config/                     # Configuration files
│   ├── default.yaml            # Main configuration
│   └── openings.json           # Chess opening positions
├── scripts/                    # Entry point scripts
│   ├── generate_data.py        # Data generation script
│   └── train.py                # Model training script
├── src/                        # Source code
│   ├── agents/                 # Chess-playing agents
│   ├── chess_env/              # Board wrapper and state encoders
│   ├── data/                   # Data generation and storage
│   ├── models/                 # Neural network architectures
│   ├── training/               # Training loop and losses
│   ├── config.py               # Configuration management
│   └── device.py               # Device selection utilities
├── requirements.txt            # Python dependencies
└── README.md                   # Quick start guide
```

---

## Dependencies

The framework requires the following Python packages:

| Package | Version | Purpose |
|---------|---------|---------|
| python-chess | >= 1.10.0 | Chess board representation and move generation |
| torch | >= 2.0.0 | Deep learning framework |
| torch-geometric | >= 2.4.0 | Graph neural network layers (GCN, GAT) |
| pyyaml | >= 6.0 | YAML configuration parsing |
| numpy | >= 1.24.0 | Numerical operations |
| tqdm | >= 4.65.0 | Progress bars |

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python scripts/generate_data.py --config config/default.yaml --num-games 100
```

### 3. Train a Model

```bash
python scripts/train.py --config config/default.yaml --epochs 10
```

---

## Architecture Overview

### Data Flow

```
Chess Board (FEN) 
    ↓
State Encoder (CNN/Transformer/GNN)
    ↓
Backbone Network
    ↓
Output Head (Policy/Value/Dual)
    ↓
Move Selection / Board Evaluation
```

### Training Pipeline

```
Stockfish Self-Play
    ↓
Position Collection (SQLite)
    ↓
Dataset Loading
    ↓
Model Training
    ↓
Checkpoint Saving
```

---

## Supported Models

| Type | Models | Edge Types |
|------|--------|------------|
| CNN | ConvNet, ResNet | N/A |
| Transformer | SquareTransformer, PieceTransformer | N/A |
| GNN | GCN, GAT | Static, Dynamic, Hybrid |

---

## Related Documentation

- [Configuration Guide](configuration.md)
- [Agents Documentation](agents.md)
- [Models Documentation](models.md)
- [Encoders Documentation](encoders.md)
- [Data Pipeline Documentation](data_pipeline.md)
- [Training Documentation](training.md)
- [Scripts Documentation](scripts.md)
