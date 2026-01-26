# Chess ML Framework Documentation

## Table of Contents

### Getting Started

- [Project Overview](project_overview.md) - Introduction, structure, and quick start guide
- [Configuration Guide](configuration.md) - YAML configuration reference
- [Scripts Documentation](scripts.md) - Command-line usage

### Core Components

- [Agents](agents.md) - Chess-playing agents (Random, UCI, Learning)
- [Encoders](encoders.md) - State encoders for CNN, Transformer, and GNN
- [Models](models.md) - Neural network architectures

### Data and Training

- [Data Pipeline](data_pipeline.md) - Database, dataset, and match runner
- [Training](training.md) - Loss functions and training loop

### Reference

- [API Reference](api_reference.md) - Quick lookup for all public APIs
- [Examples](examples.md) - Complete code examples

---

## Quick Links

### Installation

```bash
pip install -r requirements.txt
```

### Generate Training Data

```bash
python scripts/generate_data.py --num-games 100
```

### Train a Model

```bash
python scripts/train.py --model resnet --epochs 50
```

---

## Documentation Structure

```
docs/
├── index.md                 # This file
├── project_overview.md      # Project introduction
├── configuration.md         # Configuration reference
├── agents.md                # Agent documentation
├── encoders.md              # Encoder documentation
├── models.md                # Model documentation
├── data_pipeline.md         # Data pipeline documentation
├── training.md              # Training documentation
├── scripts.md               # Scripts documentation
├── api_reference.md         # API reference
└── examples.md              # Code examples
```

---

## Supported Architectures

| Type | Models | Encoder |
|------|--------|---------|
| CNN | ConvNet, ResNet | CNNEncoder |
| Transformer | SquareTransformer, PieceTransformer | TransformerEncoder |
| GNN | GCN, GAT | GNNEncoder |

---

## Version Information

- Python: >= 3.10
- PyTorch: >= 2.0.0
- torch-geometric: >= 2.4.0
