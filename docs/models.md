# Neural Network Models Documentation

## Overview

The models module (`src/models/`) contains all neural network architectures for chess move prediction. All models inherit from `ChessModel` and follow a backbone + head architecture.

---

## Architecture Design

### Backbone + Head Pattern

```
Input (Encoded Board State)
         |
    [Backbone]     <- CNN, Transformer, or GNN
         |
    Feature Vector
         |
      [Head]       <- Policy, Value, or Dual
         |
  Output (Moves/Evaluation)
```

This design allows:
- Hot-swapping backbones without changing heads
- Reusing heads across different backbones
- Easy experimentation with architecture combinations

---

## ChessModel (Base Class)

Location: `src/models/base.py`

Abstract base class for all chess neural network models.

### Abstract Methods

| Method | Description |
|--------|-------------|
| `forward_backbone(x)` | Process input and return feature vector |
| `get_backbone_output_dim()` | Return backbone output dimension |
| `name` | Property returning model name |

### Concrete Methods

| Method | Description |
|--------|-------------|
| `forward(x)` | Full forward pass through backbone and head |
| `set_head(head)` | Attach an output head to the model |
| `count_parameters()` | Count trainable parameters |

---

## CNN Models

### ConvNet

Location: `src/models/cnn/convnet.py`

Standard 6-layer Convolutional Neural Network.

#### Architecture

1. 6 convolutional blocks with increasing channels
2. Global average pooling
3. Output dimension: channels (default 256)

#### Constructor

```python
ConvNet(
    input_channels: int = 18,
    channels: int = 256,
    num_layers: int = 6,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_channels` | int | 18 | Input channels (from CNNEncoder) |
| `channels` | int | 256 | Conv layer channels |
| `num_layers` | int | 6 | Number of conv layers |

#### Model Name Format

`ConvNet_{num_layers}L_{channels}C`

Example: `ConvNet_6L_256C`

---

### ResNet

Location: `src/models/cnn/resnet.py`

ResNet-style CNN with skip connections.

#### Architecture

1. Initial convolution to project to channels
2. N residual blocks with skip connections
3. Global average pooling
4. Output dimension: channels (default 256)

#### Residual Block

```
Input
  |
  +---> Conv -> BN -> ReLU -> Conv -> BN
  |                                   |
  +-----------------------------------+
  |
  v
ReLU
```

#### Constructor

```python
ResNet(
    input_channels: int = 18,
    channels: int = 256,
    num_blocks: int = 10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_channels` | int | 18 | Input channels |
| `channels` | int | 256 | Block channels |
| `num_blocks` | int | 10 | Number of residual blocks (6, 10, or 20) |

#### Model Name Format

`ResNet_{num_blocks}B_{channels}C`

Example: `ResNet_10B_256C`

---

## Transformer Models

### SquareTransformer

Location: `src/models/transformer/square_transformer.py`

Square-based ChessFormer with 64 fixed tokens (one per square).

#### Architecture

1. Token embedding (vocabulary: 13)
2. Learnable positional embeddings (64 positions)
3. N Transformer encoder layers
4. CLS token aggregation or mean pooling
5. Output dimension: embed_dim

#### Constructor

```python
SquareTransformer(
    vocab_size: int = 13,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    use_cls_token: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 13 | Token vocabulary size |
| `embed_dim` | int | 256 | Embedding dimension |
| `num_heads` | int | 8 | Attention heads |
| `num_layers` | int | 6 | Transformer layers |
| `dropout` | float | 0.1 | Dropout rate |
| `use_cls_token` | bool | True | Use CLS token vs mean pooling |

#### Input Format

Dictionary with:
- `tokens`: Token IDs of shape (B, 64)
- `positions`: Position indices of shape (B, 64)
- `attention_mask`: Mask of shape (B, 64)
- `side_to_move`: Side to move of shape (B,)
- `castling`: Castling rights of shape (B, 4)

#### Model Name Format

`SquareTransformer_{num_layers}L_{embed_dim}D`

Example: `SquareTransformer_6L_256D`

---

### PieceTransformer

Location: `src/models/transformer/piece_transformer.py`

Piece-based ChessFormer with variable-length sequence (up to 32 pieces).

#### Architecture

1. Piece type embedding (vocabulary: 12)
2. Position embedding (64 squares)
3. Sequence position embedding
4. N Transformer encoder layers with padding mask
5. CLS token aggregation
6. Output dimension: embed_dim

#### Key Differences from SquareTransformer

- Processes only pieces, not empty squares
- Variable-length sequences (shorter in endgames)
- Potentially more efficient attention
- Uses padding masks for variable lengths

#### Constructor

```python
PieceTransformer(
    vocab_size: int = 12,
    max_pieces: int = 32,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 12 | Piece vocabulary (6 types x 2 colors) |
| `max_pieces` | int | 32 | Maximum pieces |
| `embed_dim` | int | 256 | Embedding dimension |
| `num_heads` | int | 8 | Attention heads |
| `num_layers` | int | 6 | Transformer layers |
| `dropout` | float | 0.1 | Dropout rate |

#### Model Name Format

`PieceTransformer_{num_layers}L_{embed_dim}D`

---

## GNN Models

### GCN (Graph Convolutional Network)

Location: `src/models/gnn/gcn.py`

Graph Convolutional Network for chess.

#### Architecture

1. Initial projection from node features to hidden dim
2. N GCN layers with residual connections
3. Global mean pooling over all nodes
4. Side-to-move and castling embeddings
5. Output dimension: hidden_dim

#### Edge Types

| Type | Description |
|------|-------------|
| `static` | King-move adjacency (fixed graph) |
| `dynamic` | Legal moves as edges (changes per position) |
| `hybrid` | Both edge types (heterogeneous graph) |

#### Constructor

```python
GCN(
    input_dim: int = 12,
    hidden_dim: int = 256,
    num_layers: int = 6,
    edge_type: str = "hybrid",
    dropout: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | 12 | Node feature dimension |
| `hidden_dim` | int | 256 | Hidden dimension |
| `num_layers` | int | 6 | Number of GCN layers |
| `edge_type` | str | "hybrid" | Edge construction strategy |
| `dropout` | float | 0.1 | Dropout rate |

#### Hybrid Edge Processing

For hybrid edges, the model processes spatial and legal edges separately and fuses the representations:

```
Node Features
    /         \
   /           \
Spatial GCN   Legal GCN
   \           /
    \         /
     Fusion Layer
        |
    Global Pool
```

#### Model Name Format

`GCN_{edge_type}_{num_layers}L_{hidden_dim}D`

Example: `GCN_hybrid_6L_256D`

---

### GAT (Graph Attention Network)

Location: `src/models/gnn/gat.py`

Graph Attention Network with multi-head attention.

#### Architecture

Same as GCN but uses attention-based message passing:

1. Initial projection
2. N GAT layers with multi-head attention
3. Global mean pooling
4. Side-to-move and castling embeddings
5. Output dimension: hidden_dim

#### Constructor

```python
GAT(
    input_dim: int = 12,
    hidden_dim: int = 256,
    num_layers: int = 6,
    edge_type: str = "hybrid",
    heads: int = 4,
    dropout: float = 0.1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | 12 | Node feature dimension |
| `hidden_dim` | int | 256 | Hidden dimension (divisible by heads) |
| `num_layers` | int | 6 | Number of GAT layers |
| `edge_type` | str | "hybrid" | Edge construction strategy |
| `heads` | int | 4 | Number of attention heads |
| `dropout` | float | 0.1 | Dropout rate |

#### Model Name Format

`GAT_{edge_type}_{num_layers}L_{hidden_dim}D`

---

## Output Heads

Location: `src/models/heads.py`

### PolicyHead

Outputs a probability distribution over all moves.

- **Input**: Feature vector of shape (B, input_dim)
- **Output**: Logits of shape (B, 4672)

```python
PolicyHead(input_dim: int, hidden_dim: int = 256)
```

### ValueHead

Outputs a board evaluation score.

- **Input**: Feature vector of shape (B, input_dim)
- **Output**: Value in range [-1, 1] of shape (B, 1)

```python
ValueHead(input_dim: int, hidden_dim: int = 256)
```

### DualHead

Outputs both policy and value (AlphaZero-style).

- **Input**: Feature vector of shape (B, input_dim)
- **Output**: Dictionary with `policy` and `value`

```python
DualHead(input_dim: int, hidden_dim: int = 256)
```

---

## Model Factory

Location: `src/models/factory.py`

### create_model

Creates a complete model from configuration.

```python
from src.models.factory import create_model
from src.config import load_config

config = load_config("config/default.yaml")
model = create_model(config.model)
```

### create_backbone

Creates a backbone without a head.

```python
from src.models.factory import create_backbone

backbone = create_backbone("resnet", channels=256, num_blocks=10)
```

### get_encoder_for_model

Gets the appropriate encoder for a model.

```python
from src.models.factory import get_encoder_for_model

encoder_factory = get_encoder_for_model("resnet")
encoder = encoder_factory()
```

### list_available_models

Lists all available model configurations.

```python
from src.models.factory import list_available_models

info = list_available_models()
# {'backbones': {...}, 'heads': [...], 'gnn_edge_types': [...]}
```

---

## Model Summary

| Model | Type | Encoder | Parameters (256D) |
|-------|------|---------|-------------------|
| ConvNet | CNN | CNN | ~600K |
| ResNet-10 | CNN | CNN | ~1.2M |
| ResNet-20 | CNN | CNN | ~2.4M |
| SquareTransformer | Transformer | Transformer (square) | ~2.5M |
| PieceTransformer | Transformer | Transformer (piece) | ~2.5M |
| GCN | GNN | GNN | ~1.5M |
| GAT | GNN | GNN | ~2.0M |

Note: Parameter counts are approximate and depend on head type.
