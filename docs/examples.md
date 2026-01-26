# Code Examples

## Overview

This document provides complete, runnable code examples for common tasks in the Chess ML Framework.

---

## Example 1: Basic Board Encoding

Encode a chess position for different model types.

```python
import chess
from src.chess_env.encoders import CNNEncoder, TransformerEncoder, GNNEncoder

# Create a chess board
board = chess.Board()
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")

# CNN Encoding (for ConvNet, ResNet)
cnn_encoder = CNNEncoder()
cnn_input = cnn_encoder.encode(board)
print(f"CNN input shape: {cnn_input.shape}")  # (18, 8, 8)

# Transformer Encoding (for SquareTransformer)
transformer_encoder = TransformerEncoder(tokenizer_type="square")
transformer_input = transformer_encoder.encode(board)
print(f"Transformer tokens shape: {transformer_input['tokens'].shape}")  # (64,)

# GNN Encoding (for GCN, GAT)
gnn_encoder = GNNEncoder(edge_type="hybrid")
gnn_input = gnn_encoder.encode(board)
print(f"GNN node features shape: {gnn_input['x'].shape}")  # (64, 12)
```

---

## Example 2: Create and Use a Model

Create a model from configuration and run inference.

```python
import torch
from src.config import load_config
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model
import chess

# Load configuration
config = load_config("config/default.yaml")

# Create model
model = create_model(config.model)
device = get_device()
model = model.to(device)
model.eval()

print(f"Model: {model.name}")
print(f"Parameters: {model.count_parameters():,}")

# Get encoder
encoder_factory = get_encoder_for_model(config.model.backbone)
encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

# Encode a position
board = chess.Board()
encoded = encoder.encode(board)

# Prepare input
if isinstance(encoded, torch.Tensor):
    x = encoded.unsqueeze(0).to(device)
else:
    x = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v 
         for k, v in encoded.items()}

# Run inference
with torch.no_grad():
    output = model(x)
    
print(f"Policy output shape: {output['policy'].shape}")  # (1, 4672)
if 'value' in output:
    print(f"Value output: {output['value'].item():.4f}")
```

---

## Example 3: Using the Learning Agent

Use a trained model to play chess.

```python
import torch
import chess
from src.agents import LearningAgent
from src.models.factory import create_model, get_encoder_for_model
from src.config import load_config
from src.device import get_device

# Load model
config = load_config("config/default.yaml")
model = create_model(config.model)
device = get_device()

# Load trained weights
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Create encoder
encoder_factory = get_encoder_for_model(config.model.backbone)
encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

# Create learning agent
agent = LearningAgent(
    model=model,
    encoder=encoder,
    device=device,
    temperature=0.1,  # Slight exploration
)

# Play a game
board = chess.Board()
while not board.is_game_over():
    move = agent.get_move(board)
    print(f"Move: {move}")
    board.push(move)
    
    if board.fullmove_number > 50:
        break

print(f"Final position:\n{board}")
```

---

## Example 4: Generate Training Data

Generate training data using Stockfish.

```python
from src.data import ChessDatabase, MatchRunner

# Create database
db = ChessDatabase("data/example_dataset.db")

# Create match runner
runner = MatchRunner(
    database=db,
    openings_path="config/openings.json",
    max_moves=200,
    multipv=5,
)

# Generate games
stats = runner.run_games(
    engine_path="/opt/homebrew/bin/stockfish",
    num_games=10,
    depth=15,
    save_every=5,
    verbose=True,
)

print(f"Games: {stats['num_games']}")
print(f"Positions: {stats['total_positions']}")
print(f"Results: {stats['results']}")

db.close()
```

---

## Example 5: Load and Inspect Dataset

Load training data and inspect samples.

```python
from src.data.dataset import ChessDataset, create_dataloader
from src.chess_env.encoders import CNNEncoder
import chess

# Create encoder
encoder = CNNEncoder()

# Create dataset
dataset = ChessDataset(
    db_path="data/chess_dataset.db",
    encoder=encoder,
    use_soft_labels=True,
    include_value=True,
)

print(f"Dataset size: {len(dataset)}")

# Inspect a sample
sample = dataset[0]
print(f"Input shape: {sample['input'].shape}")
print(f"Policy target shape: {sample['policy_target'].shape}")
print(f"Value target: {sample['value_target'].item():.4f}")

# Find top predicted moves
policy = sample['policy_target']
top_indices = policy.topk(5).indices
from src.chess_env.board_wrapper import INDEX_TO_UCI_MOVE
print("Top 5 moves in target:")
for idx in top_indices:
    move = INDEX_TO_UCI_MOVE.get(idx.item(), "unknown")
    prob = policy[idx].item()
    print(f"  {move}: {prob:.4f}")

# Create dataloader
loader = create_dataloader(
    db_path="data/chess_dataset.db",
    encoder=encoder,
    batch_size=32,
    shuffle=True,
)

# Iterate through batches
for batch in loader:
    print(f"Batch input shape: {batch['input'].shape}")
    print(f"Batch policy shape: {batch['policy_target'].shape}")
    break
```

---

## Example 6: Train a Model

Complete training example.

```python
from src.config import load_config
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model
from src.data.dataset import create_dataloader
from src.training import Trainer

# Configuration
config = load_config("config/default.yaml")
device = get_device()

# Create model
model = create_model(config.model)
print(f"Model: {model.name}")
print(f"Parameters: {model.count_parameters():,}")

# Create encoder and dataloader
encoder_factory = get_encoder_for_model(config.model.backbone)
encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

train_loader = create_dataloader(
    db_path=config.paths.database,
    encoder=encoder,
    batch_size=config.training.batch_size,
    shuffle=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    device=device,
    head_type=config.model.head,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    checkpoint_dir=config.paths.checkpoints,
)

# Train
history = trainer.train(
    train_loader=train_loader,
    epochs=10,
    scheduler_type="cosine",
    save_best=True,
    save_every=5,
)

print(f"Final loss: {history['train_loss'][-1]:.4f}")
```

---

## Example 7: Compare Agents

Play a game between UCI and Learning agents.

```python
import chess
from src.agents import UCIAgent, LearningAgent
from src.models.factory import create_model, get_encoder_for_model
from src.config import load_config
from src.device import get_device
import torch

# Create UCI agent (Stockfish)
uci_agent = UCIAgent(
    engine_path="/opt/homebrew/bin/stockfish",
    depth=10,
)

# Create Learning agent
config = load_config("config/default.yaml")
model = create_model(config.model)
device = get_device()

checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

encoder_factory = get_encoder_for_model(config.model.backbone)
encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

learning_agent = LearningAgent(
    model=model,
    encoder=encoder,
    device=device,
    temperature=0.0,  # Greedy
)

# Play a game
board = chess.Board()
agents = [learning_agent, uci_agent]  # Learning plays white

move_count = 0
while not board.is_game_over() and move_count < 100:
    current_agent = agents[move_count % 2]
    move = current_agent.get_move(board)
    board.push(move)
    move_count += 1

print(f"Result: {board.result()}")
print(f"Moves: {move_count}")

uci_agent.close()
```

---

## Example 8: Custom Model Configuration

Create a model with custom parameters.

```python
from src.models.cnn import ResNet
from src.models.heads import DualHead
from src.device import get_device

# Create custom ResNet
model = ResNet(
    input_channels=18,
    channels=128,      # Smaller than default
    num_blocks=6,      # Fewer blocks
)

# Attach dual head
head = DualHead(
    input_dim=model.get_backbone_output_dim(),
    hidden_dim=128,
)
model.set_head(head)

# Move to device
device = get_device()
model = model.to(device)

print(f"Model: {model.name}")
print(f"Parameters: {model.count_parameters():,}")
```

---

## Example 9: Analyze a Position

Use UCI agent to analyze a position.

```python
import chess
from src.agents import UCIAgent

# Create agent
agent = UCIAgent(
    engine_path="/opt/homebrew/bin/stockfish",
    depth=20,
    multipv=5,
)

# Set up a position
board = chess.Board()
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
board.push_san("Nc6")
board.push_san("Bb5")  # Ruy Lopez

# Get move distribution
distribution = agent.get_move_distribution(board, num_moves=5, depth=20)

print("Top 5 moves:")
for entry in distribution:
    move = board.san(entry['move'])
    score = entry['score']
    print(f"  {move}: {score} cp")

# Get best move with info
info = agent.get_move_with_info(board)
print(f"\nBest move: {board.san(info['move'])}")
print(f"Score: {info.get('score', 'N/A')} cp")
print(f"PV: {' '.join(info.get('pv', []))}")

agent.close()
```

---

## Example 10: Database Queries

Query the training database directly.

```python
from src.data import ChessDatabase

# Open database
db = ChessDatabase("data/chess_dataset.db")

# Get statistics
print(f"Total games: {db.get_game_count()}")
print(f"Total positions: {db.get_position_count()}")

# Get positions with distributions
positions = db.get_positions_with_distributions(limit=5)

for pos in positions:
    print(f"\nFEN: {pos['fen']}")
    print(f"Best move: {pos['best_move_uci']}")
    print(f"Score: {pos.get('best_move_score', 'N/A')} cp")
    
    if pos.get('move_distribution'):
        print("Distribution:")
        for move_info in pos['move_distribution'][:3]:
            print(f"  {move_info['move_uci']}: {move_info['score']} cp")

db.close()
```
