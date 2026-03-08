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

import requests
import torch

from src.config import settings
from src.data.dataset import create_dataloader
from src.models.factory import create_model, get_encoder_for_model
from src.training import Trainer

NTFY_URL = "https://ntfy.lunex.page/FYP"


def _send_ntfy(title: str, message: str, priority: str = "default") -> None:
    try:
        requests.post(
            NTFY_URL,
            data=message.encode(encoding="utf-8"),
            headers={"Title": title, "Priority": priority},
        )
    except Exception as e:
        print(f"Failed to send ntfy notification: {e}")


ALL_MODELS = [
    "convnet",
    "resnet",
    "square_transformer",
    "piece_transformer",
    "gcn",
    "gat",
]
SELF_PLAY_MODELS = ["convnet", "resnet", "square_transformer", "piece_transformer"]
STOCKFISH_RL_MODELS = [
    "convnet",
    "resnet",
    "square_transformer",
    "piece_transformer",
    "gcn",
    "gat",
]

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
        torch.set_float32_matmul_precision("high")
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
    model_overrides = {"backbone": args.model}
    if args.head:
        model_overrides["head"] = args.head
    model_cfg = settings.model.model_copy(update=model_overrides)

    training_overrides = {}
    if args.epochs:
        training_overrides["epochs"] = args.epochs
    if args.batch_size:
        training_overrides["batch_size"] = args.batch_size
    training_cfg = settings.training.model_copy(update=training_overrides)

    device = get_device(args.device)
    db_path = args.database or settings.paths.database
    checkpoint_dir = resolve_output_dir(args, "supervised", model_cfg.backbone)

    print()
    print("=" * 60)
    print("SUPERVISED TRAINING")
    if args.name:
        print(f"Run: {args.name}")
    print("=" * 60)
    print(f"Model: {model_cfg.backbone} + {model_cfg.head} head")
    print(f"Database: {db_path}")
    print(f"Epochs: {training_cfg.epochs}")
    print(f"Batch size: {training_cfg.batch_size}")
    print(f"Learning rate: {training_cfg.learning_rate}")
    print(f"Samples: {args.num_samples or 'all'}")
    print(f"Output: {checkpoint_dir}")
    print("=" * 60)

    model = create_model(model_cfg)
    print(f"\nModel: {model.name}")
    print(f"Parameters: {model.count_parameters():,}")

    encoder_factory = get_encoder_for_model(model_cfg.backbone)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()
    print(f"Encoder: {encoder.name}")

    print("\nLoading dataset...")
    train_loader = create_dataloader(
        db_path=db_path,
        encoder=encoder,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=settings.hardware.num_workers,
        include_value=(model_cfg.head in ["value", "dual"]),
        num_samples=args.num_samples,
    )
    print(f"Dataset size: {len(train_loader.dataset)} positions")

    trainer = Trainer(
        model=model,
        device=device,
        head_type=model_cfg.head,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        policy_weight=training_cfg.policy_loss_weight,
        value_weight=training_cfg.value_loss_weight,
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
        epochs=training_cfg.epochs,
        scheduler_type=training_cfg.lr_scheduler.type,
        save_best=True,
        save_every=5,
    )

    print("\nTraining complete!")


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

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Force a specific device (default: auto-detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name; organises outputs under runs/<name>/",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- supervised ---
    sp = subparsers.add_parser("supervised", help="Train on labelled database")
    sp.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    sp.add_argument(
        "--head", type=str, default=None, choices=["policy", "value", "dual"]
    )
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to .db, .parquet file, or directory of .parquet files",
    )
    sp.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit training to N positions (default: use all)",
    )
    sp.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_supervised(args)


if __name__ == "__main__":
    main()
