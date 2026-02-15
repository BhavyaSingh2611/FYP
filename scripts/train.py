#!/usr/bin/env python3
"""
Unified training script for chess models.

Supports three modes:
  supervised    - Train on labeled data from a database
  self-play     - Reinforce via MCTS self-play from an existing checkpoint
  stockfish-rl  - Reinforce by playing against Stockfish at various difficulty levels

Examples:
  python scripts/train.py supervised   --model resnet --epochs 20 --name baseline
  python scripts/train.py self-play    --model convnet --games 20 --iterations 5 --name rl_run
  python scripts/train.py stockfish-rl --model resnet --games 20 --iterations 5 --name sf_run
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import time

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.data.dataset import create_dataloader
from src.training import Trainer
from src.training.self_play import SelfPlayGenerator, games_to_tensors
from src.training.stockfish_rl import StockfishRLGenerator, stockfish_games_to_tensors


ALL_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]
SELF_PLAY_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer"]
STOCKFISH_RL_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer", "gcn", "gat"]

DIFFICULTY_LEVELS = [
    {"name": "Beginner", "skill": 0, "depth": 1, "elo": 800},
    {"name": "Novice", "skill": 1, "depth": 2, "elo": 1100},
    {"name": "Casual", "skill": 3, "depth": 3, "elo": 1400},
    {"name": "Club", "skill": 5, "depth": 5, "elo": 1700},
    {"name": "Strong", "skill": 7, "depth": 5, "elo": 2000},
]


def get_device(force: str | None = None) -> torch.device:
    """
    Select the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
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


def resolve_output_dir(args, mode: str, model_name: str) -> Path:
    """Build the output directory from --name or --output-dir."""
    if args.name:
        if mode == "supervised":
            return Path(f"runs/{args.name}/training/{model_name}")
        if mode == "stockfish_rl":
            return Path(f"runs/{args.name}/stockfish_rl/{model_name}")
        return Path(f"runs/{args.name}/self_play")
    if args.output_dir:
        return Path(args.output_dir)
    if mode == "supervised":
        return Path(f"training_results/{model_name}")
    if mode == "stockfish_rl":
        return Path(f"stockfish_rl_results/{model_name}")
    return Path("self_play_results")


# ---------------------------------------------------------------------------
# Supervised training
# ---------------------------------------------------------------------------

def run_supervised(args):
    config = load_config(args.config)

    config.model.backbone = args.model
    if args.head:
        config.model.head = args.head
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    device = get_device(args.device)
    db_path = args.database or config.paths.database
    checkpoint_dir = resolve_output_dir(args, "supervised", config.model.backbone)

    print()
    print("=" * 60)
    print("SUPERVISED TRAINING")
    if args.name:
        print(f"Run: {args.name}")
    print("=" * 60)
    print(f"Model: {config.model.backbone} + {config.model.head} head")
    print(f"Database: {db_path}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Samples: {args.num_samples or 'all'}")
    print(f"Output: {checkpoint_dir}")
    print("=" * 60)

    model = create_model(config.model)
    print(f"\nModel: {model.name}")
    print(f"Parameters: {model.count_parameters():,}")

    encoder_factory = get_encoder_for_model(config.model.backbone)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()
    print(f"Encoder: {encoder.name}")

    print("\nLoading dataset...")
    train_loader = create_dataloader(
        db_path=db_path,
        encoder=encoder,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        use_soft_labels=True,
        include_value=(config.model.head in ["value", "dual"]),
        num_samples=args.num_samples,
    )
    print(f"Dataset size: {len(train_loader.dataset)} positions")

    trainer = Trainer(
        model=model,
        device=device,
        head_type=config.model.head,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        policy_weight=config.training.policy_loss_weight,
        value_weight=config.training.value_loss_weight,
        use_soft_labels=True,
        checkpoint_dir=str(checkpoint_dir),
    )

    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from epoch {trainer.epoch}")

    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=None,
        epochs=config.training.epochs,
        scheduler_type=config.training.lr_scheduler.type,
        save_best=True,
        save_every=5,
    )

    print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Self-play training
# ---------------------------------------------------------------------------

def _build_gnn_batch(data, batch_idx, device):
    node_features = data['node_features'][batch_idx]
    batch_size = node_features.size(0)

    edge_indices = []
    edge_attrs = []
    for i, idx in enumerate(batch_idx.tolist()):
        offset = i * 64
        edge_indices.append(data['edge_indices'][idx] + offset)
        if data.get('edge_attrs') is not None:
            edge_attrs.append(data['edge_attrs'][idx])

    batch_tensor = torch.arange(batch_size, device=device)
    batch_tensor = batch_tensor.unsqueeze(1).expand(-1, 64).reshape(-1)

    result = {
        'x': node_features.view(-1, node_features.size(-1)).to(device),
        'edge_index': torch.cat(edge_indices, dim=1).to(device),
        'batch': batch_tensor,
        'side_to_move': data['side_to_move'][batch_idx].to(device),
        'castling': data['castling'][batch_idx].to(device),
    }
    if edge_attrs:
        result['edge_attr'] = torch.cat(edge_attrs, dim=0).to(device)
    return result


def train_on_self_play(model, data, device, epochs=5, batch_size=64, learning_rate=0.0001):
    model.train()

    is_dict_input = data.get('is_dict_input', False)
    is_gnn_input = data.get('is_gnn_input', False)
    n_samples = len(data['policies'])

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = {'loss': [], 'policy_loss': [], 'value_loss': []}

    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        n_batches = 0

        indices = torch.randperm(n_samples)

        for start in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            policies = data['policies'][batch_idx].to(device)
            values = data['values'][batch_idx].to(device).unsqueeze(1)

            if is_gnn_input:
                model_input = _build_gnn_batch(data, batch_idx, device)
            elif is_dict_input:
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

        print(f"  Loss: {avg_loss:.4f} (policy: {avg_policy:.4f}, value: {avg_value:.4f})")

    return history


def run_self_play(args):
    config = load_config(args.config)
    config.model.backbone = args.model
    config.model.head = "dual"

    device = get_device(args.device)
    output_dir = resolve_output_dir(args, "self_play", args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.name:
            checkpoint_path = f"runs/{args.name}/training/{args.model}/final.pt"
        else:
            checkpoint_path = f"training_results/{args.model}/final.pt"

    print()
    print("=" * 60)
    print("SELF-PLAY TRAINING")
    if args.name:
        print(f"Run: {args.name}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Games per iteration: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Training epochs: {args.epochs or 10}")
    print(f"Iterations: {args.iterations}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    model = create_model(config.model)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters():,}")

    encoder_factory = get_encoder_for_model(args.model)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

    if args.dry_run:
        print("\n[DRY RUN] Testing self-play generation...")
        generator = SelfPlayGenerator(
            model=model, encoder=encoder, device=device, num_simulations=10,
        )
        game = generator.generate_game(max_moves=10)
        print(f"Generated test game: {len(game.moves)} moves, result: {game.result}")
        print("Dry run successful!")
        return

    epochs = args.epochs or 10
    all_history = []

    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")

        print(f"\nGenerating {args.games} self-play games...")
        generator = SelfPlayGenerator(
            model=model, encoder=encoder, device=device,
            num_simulations=args.simulations,
        )

        start_time = time.time()
        games = generator.generate_games(args.games)
        gen_time = time.time() - start_time

        total_positions = sum(g.num_positions for g in games)
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        for g in games:
            results[g.result] = results.get(g.result, 0) + 1

        print(f"Generated {total_positions} positions in {gen_time:.1f}s")
        print(f"Results: W:{results['1-0']} B:{results['0-1']} D:{results['1/2-1/2']}")

        print("\nConverting to training data...")
        data = games_to_tensors(games, encoder, model_type=args.model)
        n_examples = len(data.get('inputs', data.get('tokens', [])))
        print(f"Training data: {n_examples} examples")

        print(f"\nTraining for {epochs} epochs...")
        history = train_on_self_play(
            model=model, data=data, device=device, epochs=epochs,
        )

        all_history.append({
            'iteration': iteration + 1,
            'games': args.games,
            'positions': total_positions,
            'results': results,
            'history': history,
        })

        ckpt_name = f"{args.model}_selfplay_iter{iteration + 1}.pt"
        ckpt_path = output_dir / ckpt_name
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model.name,
            'iteration': iteration + 1,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / f"{args.model}_selfplay_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.name,
        'total_iterations': args.iterations,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    log_path = output_dir / f"{args.model}_selfplay_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'model': args.model,
            'config': {
                'games_per_iter': args.games,
                'simulations': args.simulations,
                'epochs': epochs,
                'iterations': args.iterations,
            },
            'history': all_history,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Training log saved: {log_path}")

    print("\n" + "=" * 60)
    print("SELF-PLAY TRAINING COMPLETE!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Stockfish RL training
# ---------------------------------------------------------------------------

def run_stockfish_rl(args):
    config = load_config(args.config)
    config.model.backbone = args.model
    config.model.head = "dual"

    device = get_device(args.device)
    output_dir = resolve_output_dir(args, "stockfish_rl", args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.name:
            checkpoint_path = f"runs/{args.name}/training/{args.model}/final.pt"
        else:
            checkpoint_path = f"training_results/{args.model}/final.pt"

    stockfish_path = args.stockfish_path or config.engines["stockfish"].path
    levels = [l for l in DIFFICULTY_LEVELS if l["skill"] <= args.max_skill]
    if not levels:
        levels = [DIFFICULTY_LEVELS[0]]
    epochs = args.epochs or 3

    print()
    print("=" * 60)
    print("STOCKFISH RL TRAINING")
    if args.name:
        print(f"Run: {args.name}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Stockfish: {stockfish_path}")
    print(f"Games per iteration: {args.games}")
    print(f"Training epochs: {epochs}")
    print(f"Iterations: {args.iterations}")
    print(f"Max skill level: {args.max_skill}")
    print(f"Eval depth: {args.eval_depth}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Difficulty levels: {', '.join(l['name'] for l in levels)}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    model = create_model(config.model)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict']
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters():,}")

    encoder_factory = get_encoder_for_model(args.model)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()

    if args.dry_run:
        print("\n[DRY RUN] Testing Stockfish RL generation...")
        level = levels[0]
        generator = StockfishRLGenerator(
            model=model, encoder=encoder, device=device,
            stockfish_path=stockfish_path, skill_level=level['skill'],
            opponent_depth=level['depth'], eval_depth=args.eval_depth,
            temperature=0.5,
        )
        game = generator.generate_game(max_moves=10)
        print(f"Generated test game: {len(game.moves)} moves, result: {game.result}")
        print("Dry run successful!")
        return

    all_history = []

    for iteration in range(args.iterations):
        if args.curriculum:
            level_idx = min(iteration, len(levels) - 1)
        else:
            level_idx = iteration % len(levels)
        level = levels[level_idx]

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.iterations} — {level['name']} (skill={level['skill']}, depth={level['depth']}, ~{level['elo']} ELO)")
        print(f"{'='*60}")

        print(f"\nGenerating {args.games} games vs Stockfish...")
        generator = StockfishRLGenerator(
            model=model, encoder=encoder, device=device,
            stockfish_path=stockfish_path, skill_level=level['skill'],
            opponent_depth=level['depth'], eval_depth=args.eval_depth,
            temperature=0.5,
        )

        start_time = time.time()
        games = generator.generate_games(args.games)
        gen_time = time.time() - start_time

        total_positions = sum(g.num_positions for g in games)
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        for g in games:
            results[g.result] = results.get(g.result, 0) + 1

        print(f"Generated {total_positions} positions in {gen_time:.1f}s")
        print(f"Results: W:{results['1-0']} B:{results['0-1']} D:{results['1/2-1/2']}")

        print("\nConverting to training data...")
        data = stockfish_games_to_tensors(games, encoder, model_type=args.model)
        n_examples = len(data.get('inputs', data.get('tokens', [])))
        print(f"Training data: {n_examples} examples")

        print(f"\nTraining for {epochs} epochs...")
        history = train_on_self_play(
            model=model, data=data, device=device,
            epochs=epochs, learning_rate=5e-5,
        )

        all_history.append({
            'iteration': iteration + 1,
            'level': level,
            'games': args.games,
            'positions': total_positions,
            'results': results,
            'history': history,
        })

        ckpt_name = f"{args.model}_stockfish_iter{iteration + 1}.pt"
        ckpt_path = output_dir / ckpt_name
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model.name,
            'iteration': iteration + 1,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / f"{args.model}_stockfish_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.name,
        'total_iterations': args.iterations,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    log_path = output_dir / f"{args.model}_stockfish_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'model': args.model,
            'config': {
                'games_per_iter': args.games,
                'epochs': epochs,
                'iterations': args.iterations,
                'max_skill': args.max_skill,
                'eval_depth': args.eval_depth,
                'curriculum': args.curriculum,
            },
            'history': all_history,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Training log saved: {log_path}")

    print("\n" + "=" * 60)
    print("STOCKFISH RL TRAINING COMPLETE!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train chess models (supervised, self-play, or stockfish-rl)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/train.py supervised   --model resnet --epochs 20 --name baseline\n"
            "  python scripts/train.py self-play    --model convnet --games 20 --name rl_run\n"
            "  python scripts/train.py stockfish-rl --model resnet --games 20 --name sf_run\n"
        ),
    )

    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Force a specific device (default: auto-detect)")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name; organises outputs under runs/<name>/")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- supervised ---
    sp = subparsers.add_parser("supervised", help="Train on labelled database")
    sp.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    sp.add_argument("--head", type=str, default=None, choices=["policy", "value", "dual"])
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--database", type=str, default=None,
                    help="Path to .db, .parquet file, or directory of .parquet files")
    sp.add_argument("--num-samples", type=int, default=None,
                    help="Limit training to N positions (default: use all)")
    sp.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from checkpoint")

    # --- self-play ---
    sp2 = subparsers.add_parser("self-play", help="Reinforce via MCTS self-play")
    sp2.add_argument("--model", type=str, required=True, choices=SELF_PLAY_MODELS)
    sp2.add_argument("--checkpoint", type=str, default=None,
                     help="Checkpoint to start from (default: auto-resolve from --name or training_results/)")
    sp2.add_argument("--games", type=int, default=20,
                     help="Self-play games per iteration")
    sp2.add_argument("--simulations", type=int, default=100,
                     help="MCTS simulations per move")
    sp2.add_argument("--epochs", type=int, default=None,
                     help="Training epochs per iteration (default: 10)")
    sp2.add_argument("--iterations", type=int, default=1,
                     help="Number of generate-then-train iterations")
    sp2.add_argument("--dry-run", action="store_true",
                     help="Quick test without actual training")

    # --- stockfish-rl ---
    sp3 = subparsers.add_parser("stockfish-rl", help="Reinforce by playing against Stockfish")
    sp3.add_argument("--model", type=str, required=True, choices=STOCKFISH_RL_MODELS)
    sp3.add_argument("--checkpoint", type=str, default=None)
    sp3.add_argument("--games", type=int, default=20, help="Games per iteration")
    sp3.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    sp3.add_argument("--epochs", type=int, default=None, help="Training epochs per iteration (default: 3)")
    sp3.add_argument("--max-skill", type=int, default=3, help="Max Stockfish skill level to train against")
    sp3.add_argument("--eval-depth", type=int, default=8, help="Stockfish depth for reward evaluation")
    sp3.add_argument("--curriculum", action="store_true", help="Start from easiest level and progress")
    sp3.add_argument("--stockfish-path", type=str, default=None,
                     help="Override Stockfish engine path (default: from config)")
    sp3.add_argument("--dry-run", action="store_true", help="Quick test")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "supervised":
        run_supervised(args)
    elif args.mode == "self-play":
        run_self_play(args)
    elif args.mode == "stockfish-rl":
        run_stockfish_rl(args)


if __name__ == "__main__":
    main()
