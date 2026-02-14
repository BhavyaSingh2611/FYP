#!/usr/bin/env python3
"""
Run self-play reinforcement learning for all six model architectures.

Loads supervised checkpoints from runs/<run_name>/ and runs MCTS self-play
training with architecture-specific hyperparameters tuned to each model's
size, inference speed, and learning characteristics.

Examples:
  python scripts/self_play_all.py --run 50_10M --games 20 --iterations 3
  python scripts/self_play_all.py --run 50_10M --models resnet convnet --dry-run
  python scripts/self_play_all.py --run 50_10M --games 10 --simulations 50 --device mps
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import chess
import requests
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.training.self_play import SelfPlayGenerator, SelfPlayGame
from src.chess_env.board_wrapper import UCI_MOVE_TO_INDEX


ALL_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]

NTFY_URL = "https://ntfy.lunex.page/FYP"


def _send_ntfy(title: str, message: str, priority: str = "default") -> None:
    try:
        requests.post(NTFY_URL,
            data=message.encode(encoding='utf-8'),
            headers={"Title": title, "Priority": priority})
    except Exception as e:
        print(f"Failed to send ntfy notification: {e}")

# Per-model self-play hyperparameters.
#
# Rationale:
#   - simulations: Larger models (resnet, transformers) need fewer MCTS sims
#     because their policy/value estimates are already stronger. Smaller/weaker
#     models (convnet, GNNs) benefit from more search.
#   - epochs: More training epochs per iteration for small models so they can
#     absorb the self-play signal. Large models risk overfitting on small
#     self-play batches so fewer epochs.
#   - batch_size: Matched to model size — larger models use smaller batches
#     to fit in memory; small models can use larger batches.
#   - learning_rate: Conservative for large models to avoid catastrophic
#     forgetting. Slightly higher for small models that need stronger updates.
#   - temperature_moves: How many opening moves use temperature sampling for
#     diversity. Larger for models that already play well to keep exploring.
#   - c_puct: Exploration constant. Higher for weaker models to encourage
#     broader search.
MODEL_CONFIGS = {
    "convnet": {
        # 4.3M params — lightweight CNN, fast inference
        "simulations": 100,
        "epochs": 10,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "temperature_moves": 20,
        "c_puct": 1.5,
    },
    "resnet": {
        # 13.2M params — largest model, strong policy/value
        "simulations": 80,
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 5e-5,
        "temperature_moves": 30,
        "c_puct": 1.4,
    },
    "square_transformer": {
        # 6.1M params — attention-based, medium cost
        "simulations": 80,
        "epochs": 8,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "temperature_moves": 25,
        "c_puct": 1.4,
    },
    "piece_transformer": {
        # 6.1M params — similar to square transformer
        "simulations": 80,
        "epochs": 8,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "temperature_moves": 25,
        "c_puct": 1.4,
    },
    "gcn": {
        # 1.8M params — small GNN, fast but weaker
        "simulations": 120,
        "epochs": 12,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "temperature_moves": 20,
        "c_puct": 1.6,
    },
    "gat": {
        # 1.9M params — small GNN with attention
        "simulations": 120,
        "epochs": 12,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "temperature_moves": 20,
        "c_puct": 1.6,
    },
}


def get_device(force: str | None = None) -> torch.device:
    if force and force != "auto":
        dev = torch.device(force)
        print(f"Device: {dev} (forced)")
        return dev
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Device: cuda ({name})")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Device: mps (Apple Silicon)")
        return torch.device("mps")
    print("Device: cpu")
    return torch.device("cpu")


def resolve_checkpoint(run_name: str, model_name: str) -> Path:
    """Locate the supervised checkpoint for a model."""
    candidates = [
        Path(f"runs/{run_name}/{model_name}.pt"),
        Path(f"runs/{run_name}/training/{model_name}/final.pt"),
        Path(f"runs/{run_name}/training/{model_name}/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found for {model_name} in runs/{run_name}/. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Self-play data conversion (extended to support GNN graphs)
# ---------------------------------------------------------------------------

def games_to_tensors(games: list[SelfPlayGame], encoder, model_type: str) -> dict:
    """Convert self-play games to training tensors, supporting all encoders."""
    all_policies = []
    all_values = []

    is_gnn = model_type in ["gcn", "gat"]
    is_transformer = model_type in ["square_transformer", "piece_transformer"]

    # CNN
    all_inputs = []
    # Transformer
    all_tokens = []
    all_positions = []
    all_attention_masks = []
    all_side_to_move = []
    all_castling = []
    # GNN (variable-size edges → keep as lists)
    all_node_features = []
    all_edge_indices = []
    all_edge_attrs = []
    all_gnn_side = []
    all_gnn_castling = []

    for game in games:
        for example in game.examples:
            board = chess.Board(example.fen)
            encoded = encoder.encode(board)

            if is_gnn:
                all_node_features.append(encoded['x'])
                all_edge_indices.append(encoded['edge_index'])
                all_edge_attrs.append(encoded.get('edge_attr'))
                all_gnn_side.append(encoded['side_to_move'])
                all_gnn_castling.append(encoded['castling'])
            elif isinstance(encoded, dict):
                all_tokens.append(encoded['tokens'])
                all_positions.append(encoded['positions'])
                all_attention_masks.append(encoded['attention_mask'])
                all_side_to_move.append(encoded['side_to_move'])
                all_castling.append(encoded['castling'])
            else:
                all_inputs.append(encoded)

            policy = torch.zeros(len(UCI_MOVE_TO_INDEX))
            for uci, prob in example.policy.items():
                idx = UCI_MOVE_TO_INDEX.get(uci, -1)
                if idx >= 0:
                    policy[idx] = prob
            all_policies.append(policy)
            all_values.append(example.value)

    result = {
        'policies': torch.stack(all_policies),
        'values': torch.tensor(all_values, dtype=torch.float32),
        'model_type': model_type,
    }

    if is_gnn:
        result['x'] = torch.stack(all_node_features)
        result['edge_index'] = all_edge_indices  # list — variable size
        result['edge_attr'] = all_edge_attrs
        result['side_to_move'] = torch.stack(all_gnn_side) if all_gnn_side else None
        result['castling'] = torch.stack(all_gnn_castling) if all_gnn_castling else None
    elif is_transformer and all_tokens:
        result['tokens'] = torch.stack(all_tokens)
        result['positions'] = torch.stack(all_positions)
        result['attention_mask'] = torch.stack(all_attention_masks)
        result['side_to_move'] = torch.stack(all_side_to_move)
        result['castling'] = torch.stack(all_castling)
    elif all_inputs:
        result['inputs'] = torch.stack(all_inputs)

    return result


# ---------------------------------------------------------------------------
# Training loop (handles all three input types)
# ---------------------------------------------------------------------------

def train_on_self_play(
    model, data: dict, device: torch.device,
    epochs: int, batch_size: int, learning_rate: float,
) -> dict:
    model.train()

    model_type = data.get('model_type', 'cnn')
    is_gnn = model_type in ["gcn", "gat"]
    is_transformer = model_type in ["square_transformer", "piece_transformer"]
    n_samples = len(data['policies'])

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = {'loss': [], 'policy_loss': [], 'value_loss': []}

    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        n_batches = 0

        indices = torch.randperm(n_samples)

        for start in tqdm(range(0, n_samples, batch_size), desc=f"  Epoch {epoch+1}/{epochs}"):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            policies = data['policies'][batch_idx].to(device)
            values = data['values'][batch_idx].to(device).unsqueeze(1)

            if is_gnn:
                model_input = _build_gnn_batch(data, batch_idx, device)
            elif is_transformer:
                model_input = {
                    'tokens': data['tokens'][batch_idx].to(device),
                    'positions': data['positions'][batch_idx].to(device),
                    'attention_mask': data['attention_mask'][batch_idx].to(device),
                    'side_to_move': data['side_to_move'][batch_idx].to(device),
                    'castling': data['castling'][batch_idx].to(device),
                }
            else:
                model_input = data['inputs'][batch_idx].to(device)

            optimizer.zero_grad()
            output = model(model_input)

            policy_logits = output['policy']
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(policies * log_probs).sum(dim=-1).mean()

            value_pred = output['value']
            value_loss = F.mse_loss(value_pred, values)

            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_policy = policy_loss_sum / n_batches
        avg_value = value_loss_sum / n_batches

        history['loss'].append(avg_loss)
        history['policy_loss'].append(avg_policy)
        history['value_loss'].append(avg_value)

        print(f"    Loss: {avg_loss:.4f} (policy: {avg_policy:.4f}, value: {avg_value:.4f})")

    return history


def _build_gnn_batch(data: dict, batch_idx: torch.Tensor, device: torch.device) -> dict:
    """Build a batched GNN input dict compatible with PyTorch Geometric."""
    node_features = data['x'][batch_idx].to(device)
    edge_indices = []
    for i, idx in enumerate(batch_idx):
        edge_indices.append(data['edge_index'][idx.item()] + i * 64)
    edge_index = torch.cat(edge_indices, dim=1).to(device)
    batch = torch.arange(len(batch_idx), device=device).unsqueeze(1).expand(-1, 64).reshape(-1)

    result = {
        'x': node_features.view(-1, node_features.size(-1)),
        'edge_index': edge_index,
        'batch': batch,
        'side_to_move': data['side_to_move'][batch_idx].to(device),
        'castling': data['castling'][batch_idx].to(device),
    }
    return result


# ---------------------------------------------------------------------------
# Per-model self-play runner
# ---------------------------------------------------------------------------

def run_model_self_play(
    model_name: str,
    run_name: str,
    device: torch.device,
    num_games: int,
    num_iterations: int,
    override_sims: int | None,
    dry_run: bool,
) -> dict | None:
    """Run full self-play loop for a single model."""
    cfg = MODEL_CONFIGS[model_name]
    simulations = override_sims or cfg["simulations"]

    config = load_config("config/default.yaml")
    config.model.backbone = model_name
    config.model.head = "dual"

    checkpoint_path = resolve_checkpoint(run_name, model_name)

    output_dir = Path(f"runs/{run_name}/self_play/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(f"SELF-PLAY: {model_name.upper()}")
    print("=" * 60)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Simulations/move: {simulations}")
    print(f"Games/iteration:  {num_games}")
    print(f"Iterations:       {num_iterations}")
    print(f"Epochs/iter:      {cfg['epochs']}")
    print(f"Batch size:       {cfg['batch_size']}")
    print(f"Learning rate:    {cfg['learning_rate']}")
    print(f"C_puct:           {cfg['c_puct']}")
    print(f"Temp moves:       {cfg['temperature_moves']}")
    print(f"Output:           {output_dir}")
    print("=" * 60)

    model = create_model(config.model)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Parameters: {model.count_parameters():,}")

    encoder_factory = get_encoder_for_model(model_name)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

    if dry_run:
        print("\n[DRY RUN] Testing self-play generation...")
        generator = SelfPlayGenerator(
            model=model, encoder=encoder, device=device, num_simulations=10,
        )
        game = generator.generate_game(max_moves=10)
        print(f"Generated test game: {len(game.moves)} moves, result: {game.result}")
        print("Dry run successful!")
        return None

    all_history = []

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        generator = SelfPlayGenerator(
            model=model, encoder=encoder, device=device,
            num_simulations=simulations,
            c_puct=cfg["c_puct"],
            temperature_moves=cfg["temperature_moves"],
        )

        start_time = time.time()
        games = generator.generate_games(num_games)
        gen_time = time.time() - start_time

        total_positions = sum(g.num_positions for g in games)
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        for g in games:
            results[g.result] = results.get(g.result, 0) + 1

        print(f"  Generated {total_positions} positions in {gen_time:.1f}s")
        print(f"  Results: W:{results['1-0']} B:{results['0-1']} D:{results['1/2-1/2']}")

        data = games_to_tensors(games, encoder, model_type=model_name)

        history = train_on_self_play(
            model=model, data=data, device=device,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
        )

        all_history.append({
            'iteration': iteration + 1,
            'games': num_games,
            'positions': total_positions,
            'results': results,
            'generation_time_s': gen_time,
            'history': history,
        })

        ckpt_path = output_dir / f"iter{iteration + 1}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model.name,
            'iteration': iteration + 1,
            'config': cfg,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        _send_ntfy(
            title=f"Self-Play {model_name} - Iter {iteration+1}/{num_iterations}",
            message=f"Games: {num_games} | Positions: {total_positions}\n"
                    f"Loss: {history['loss'][-1]:.4f} (P: {history['policy_loss'][-1]:.4f}, V: {history['value_loss'][-1]:.4f})\n"
                    f"Results: W:{results['1-0']} B:{results['0-1']} D:{results['1/2-1/2']}",
        )

    final_path = output_dir / "final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.name,
        'total_iterations': num_iterations,
        'config': cfg,
    }, final_path)

    log_path = output_dir / "self_play_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'model': model_name,
            'run': run_name,
            'config': cfg,
            'history': all_history,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n  Final model: {final_path}")
    print(f"  Log:         {log_path}")

    final_loss = all_history[-1]['history']['loss'][-1] if all_history else None
    _send_ntfy(
        title=f"Self-Play {model_name} - Complete ✓",
        message=f"Finished {num_iterations} iterations ({num_games} games each)\n"
                f"Final loss: {final_loss:.4f}" if final_loss else "No training data",
        priority="high",
    )

    return {
        'model': model_name,
        'iterations': num_iterations,
        'final_loss': all_history[-1]['history']['loss'][-1] if all_history else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-play training for all model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/self_play_all.py --run 50_10M --games 20 --iterations 3\n"
            "  python scripts/self_play_all.py --run 50_10M --models resnet convnet\n"
            "  python scripts/self_play_all.py --run 50_10M --dry-run\n"
        ),
    )
    parser.add_argument("--run", type=str, required=True,
                        help="Run name containing supervised checkpoints (e.g. 50_10M)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Models to train (default: all six)")
    parser.add_argument("--games", type=int, default=50,
                        help="Self-play games per iteration (default: 50)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of generate-then-train iterations (default: 5)")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Override MCTS simulations per move (default: per-model)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: generate one short game per model, no training")

    args = parser.parse_args()
    device = get_device(args.device)

    print("\n" + "=" * 60)
    print("SELF-PLAY TRAINING — ALL MODELS")
    print("=" * 60)
    print(f"Run:        {args.run}")
    print(f"Models:     {', '.join(args.models)}")
    print(f"Games/iter: {args.games}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)

    summary = []
    for model_name in args.models:
        try:
            result = run_model_self_play(
                model_name=model_name,
                run_name=args.run,
                device=device,
                num_games=args.games,
                num_iterations=args.iterations,
                override_sims=args.simulations,
                dry_run=args.dry_run,
            )
            if result:
                summary.append(result)
        except FileNotFoundError as e:
            print(f"\n⚠ Skipping {model_name}: {e}")
        except Exception as e:
            print(f"\n✗ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if summary:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for s in summary:
            loss_str = f"{s['final_loss']:.4f}" if s['final_loss'] else "N/A"
            print(f"  {s['model']:25s}  iters={s['iterations']}  final_loss={loss_str}")
        print("=" * 60)


if __name__ == "__main__":
    main()
