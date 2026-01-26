# State Encoders Documentation

## Overview

The encoders module (`src/chess_env/encoders/`) provides architecture-specific board state encodings. Each encoder transforms a `chess.Board` into a format suitable for its corresponding neural network architecture.

---

## StateEncoder (Base Class)

Location: `src/chess_env/encoders/base.py`

Abstract base class defining the encoder interface.

### Abstract Methods

| Method | Description |
|--------|-------------|
| `encode(board)` | Encode a chess board state |
| `get_input_shape()` | Get the shape of the encoded input |
| `name` | Property returning the encoder name |

---

## CNNEncoder

Location: `src/chess_env/encoders/cnn_encoder.py`

Encodes the chess board as an 8x8xN tensor (bitboard representation) for CNN processing.

### Output Format

- **Shape**: `(18, 8, 8)` tensor
- **Type**: `torch.Tensor`

### Channel Layout

| Channels | Content |
|----------|---------|
| 0-5 | Our pieces (pawn, knight, bishop, rook, queen, king) |
| 6-11 | Their pieces (pawn, knight, bishop, rook, queen, king) |
| 12 | Our kingside castling rights (filled plane) |
| 13 | Our queenside castling rights (filled plane) |
| 14 | Their kingside castling rights (filled plane) |
| 15 | Their queenside castling rights (filled plane) |
| 16 | En passant square (single square marked) |
| 17 | Side to move (filled plane) |

### Perspective Handling

By default, the board is always encoded from the current player's perspective:
- "Our" pieces are in channels 0-5
- "Their" pieces are in channels 6-11
- Board is flipped when encoding for Black

### Constructor

```python
CNNEncoder(flip_perspective: bool = True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flip_perspective` | bool | True | Always encode from current player's perspective |

### Example Usage

```python
from src.chess_env.encoders import CNNEncoder
import chess

encoder = CNNEncoder()
board = chess.Board()

# Encode board
tensor = encoder.encode(board)  # Shape: (18, 8, 8)

# Get input shape
shape = encoder.get_input_shape()  # (18, 8, 8)
```

### Utility Methods

#### decode_piece_planes

```python
pieces = encoder.decode_piece_planes(tensor)
# Returns: {'our_pawns': ['e2', 'e4'], 'our_knights': ['g1', 'f3'], ...}
```

Decodes piece positions from a tensor for debugging.

---

## TransformerEncoder

Location: `src/chess_env/encoders/transformer_encoder.py`

Encodes the chess board as a token sequence for Transformer processing. Supports two tokenization schemes.

### Tokenization Schemes

#### Square Tokenizer (64 Fixed Tokens)

Each of the 64 squares gets one token, regardless of whether a piece is present.

- **Vocabulary Size**: 13 (empty + 12 piece types)
- **Sequence Length**: 64 (fixed)
- **Token Values**: 0=empty, 1-6=white pieces, 7-12=black pieces

#### Piece Tokenizer (Variable-Length)

Only pieces on the board get tokens, not empty squares.

- **Vocabulary Size**: 12 (6 piece types x 2 colors)
- **Sequence Length**: Up to 32 (maximum pieces on board)
- **Includes**: Piece type, square position, attention mask

### Output Format

Returns a dictionary containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `tokens` | (64,) or (32,) | Token IDs |
| `positions` | (64,) or (32,) | Position indices |
| `attention_mask` | (64,) or (32,) | Valid positions mask |
| `side_to_move` | (1,) | 0 for white, 1 for black |
| `castling` | (4,) | Castling rights |

### Constructor

```python
TransformerEncoder(tokenizer_type: str = "square")
```

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `tokenizer_type` | str | "square" | "square", "piece" | Tokenization scheme |

### Example Usage

```python
from src.chess_env.encoders import TransformerEncoder
import chess

# Square tokenizer
encoder = TransformerEncoder(tokenizer_type="square")
board = chess.Board()

encoded = encoder.encode(board)
# encoded['tokens'].shape: (64,)
# encoded['positions'].shape: (64,)
# encoded['attention_mask'].shape: (64,)
# encoded['side_to_move']: tensor(0)
# encoded['castling'].shape: (4,)

# Piece tokenizer
encoder = TransformerEncoder(tokenizer_type="piece")
encoded = encoder.encode(board)
# encoded['tokens'].shape: (32,)  # Max pieces
# encoded['num_pieces']: 32       # Actual piece count
```

### Properties

| Property | Description |
|----------|-------------|
| `vocab_size` | Vocabulary size (13 for square, 12 for piece) |
| `max_seq_length` | Maximum sequence length (64 for square, 32 for piece) |

---

## GNNEncoder

Location: `src/chess_env/encoders/gnn_encoder.py`

Encodes the chess board as a graph for Graph Neural Network processing.

### Graph Structure

- **Nodes**: 64 nodes (one per square)
- **Node Features**: 12-dimensional feature vector
- **Edges**: Configurable based on edge type

### Node Feature Layout

| Index | Feature | Values |
|-------|---------|--------|
| 0 | Empty square | 1 if empty |
| 1-6 | Piece type one-hot | Pawn through King |
| 7 | Piece color | 0=white, 0.5=empty, 1=black |
| 8 | Row position | Normalized [0, 1] |
| 9 | Column position | Normalized [0, 1] |
| 10 | Attacked by us | 1 if attacked |
| 11 | Attacked by them | 1 if attacked |

### Edge Types

#### Static Edges

King-move adjacency pattern (8-connectivity). Creates a fixed graph structure that does not change during the game.

- Each square connected to up to 8 neighbors
- Same edges for all positions

#### Dynamic Edges

Current legal moves as directed edges. Captures tactical possibilities in the position.

- Edge from source square to target square for each legal move
- Changes every position

#### Hybrid Edges

Both static and dynamic edges as a heterogeneous graph.

- `spatial`: King-move adjacency edges
- `legal`: Current legal move edges

### Output Format

Returns a dictionary containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `x` | (64, 12) | Node features |
| `edge_index` | (2, E) or dict | Edge indices |
| `side_to_move` | (1,) | 0 for white, 1 for black |
| `castling` | (4,) | Castling rights |

### Constructor

```python
GNNEncoder(edge_type: str = "hybrid")
```

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `edge_type` | str | "hybrid" | "static", "dynamic", "hybrid" | Edge construction strategy |

### Example Usage

```python
from src.chess_env.encoders import GNNEncoder
import chess

# Static edges (fixed graph)
encoder = GNNEncoder(edge_type="static")
board = chess.Board()

encoded = encoder.encode(board)
# encoded['x'].shape: (64, 12)
# encoded['edge_index'].shape: (2, 420)  # ~420 static edges

# Dynamic edges (legal moves)
encoder = GNNEncoder(edge_type="dynamic")
encoded = encoder.encode(board)
# encoded['edge_index'].shape: (2, 20)  # ~20 legal moves

# Hybrid edges (both)
encoder = GNNEncoder(edge_type="hybrid")
encoded = encoder.encode(board)
# encoded['edge_index']['spatial'].shape: (2, 420)
# encoded['edge_index']['legal'].shape: (2, 20)
```

---

## BoardWrapper

Location: `src/chess_env/board_wrapper.py`

Wrapper around `python-chess.Board` with additional functionality.

### Move Space Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `ALL_UCI_MOVES` | list | All possible UCI move strings |
| `UCI_MOVE_TO_INDEX` | dict | UCI string to index mapping |
| `INDEX_TO_UCI_MOVE` | dict | Index to UCI string mapping |
| `NUM_MOVES` | 4672 | Total number of possible moves |

### Key Methods

#### Board State

```python
wrapper = BoardWrapper()
wrapper.fen           # Current FEN string
wrapper.turn          # True for White, False for Black
wrapper.is_game_over  # Game termination check
wrapper.result        # Game result string
wrapper.legal_moves   # List of legal Move objects
wrapper.legal_moves_uci  # List of legal moves as UCI strings
```

#### Move Manipulation

```python
wrapper.push("e2e4")        # Make a move (UCI string or Move)
wrapper.pop()               # Undo last move
wrapper.reset()             # Reset to starting position
wrapper.set_fen(fen_string) # Set position from FEN
```

#### Move Index Conversion

```python
# Convert move to policy index
move_idx = BoardWrapper.move_to_index("e2e4")  # Returns index 0-4671

# Convert index back to UCI string  
move_uci = BoardWrapper.index_to_move(move_idx)  # Returns "e2e4"
```

---

## Encoder Selection

| Model Type | Encoder | Notes |
|------------|---------|-------|
| ConvNet | CNNEncoder | 18-channel bitboard |
| ResNet | CNNEncoder | 18-channel bitboard |
| SquareTransformer | TransformerEncoder (square) | 64 fixed tokens |
| PieceTransformer | TransformerEncoder (piece) | Variable-length tokens |
| GCN | GNNEncoder | Graph with configurable edges |
| GAT | GNNEncoder | Graph with configurable edges |

Use `get_encoder_for_model()` from `src/models/factory.py` to automatically get the correct encoder for a model backbone.
