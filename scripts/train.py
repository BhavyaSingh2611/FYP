#!/usr/bin/env python3
"""
Training script for chess models (supervised).

Trains continuously by default (omit --epochs). A live control API
starts on :5050 for adjusting hyperparameters mid-run.
Stats are saved to <output>/training_stats.jsonl.

Examples:
  python scripts/train.py --model resnet --name baseline
  python scripts/train.py --model resnet --epochs 20 --name baseline
"""

import argparse
import logging
from pathlib import Path

from src.config import settings
from src.data.dataset import create_dataloader
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model
from src.training import Trainer, start_control_server

LOGGER = logging.getLogger(__name__)

ALL_MODELS = [
    "convnet",
    "resnet",
    "square_transformer",
    "piece_transformer",
    "gcn",
    "gat",
]


def resolve_output_dir(args, model_name: str) -> Path:
    """Build the output directory from --name or --output-dir."""
    if args.name:
        return Path(f"runs/{args.name}/training/{model_name}")
    if args.output_dir:
        return Path(args.output_dir)
    return Path(f"training_results/{model_name}")


def run_training(args):
    model_overrides = {}

    if args.head:
        model_overrides["head"] = args.head

    model_cfg = settings.model.model_copy(update=model_overrides) if model_overrides else settings.model

    training_overrides = {}

    if args.epochs:
        training_overrides["epochs"] = args.epochs
    if args.batch_size:
        training_overrides["batch_size"] = args.batch_size
    if args.save_every:
        training_overrides["save_every"] = args.save_every

    training_cfg = settings.training.model_copy(update=training_overrides)

    device = get_device()
    db_path = args.database or settings.paths.database
    checkpoint_dir = resolve_output_dir(args, args.model)

    model = create_model(args.model, model_cfg)

    continuous = args.continuous or args.epochs is None
    epoch_display = "∞ (continuous)" if continuous else str(training_cfg.epochs)

    banner = f"""\
        {f"Run: {args.name}" if args.name else ""}
        {"-" * 60}
        Model:         {args.model} + {model_cfg.head} head ({model.name})
        Parameters:    {model.count_parameters():,}
        Database:      {db_path}
        Epochs:        {epoch_display}
        Batch size:    {training_cfg.batch_size}
        Learning rate: {training_cfg.learning_rate}
        Samples:       {args.num_samples or "all"}
        Output:        {checkpoint_dir}
        Control API:   http://0.0.0.0:{args.control_port}
        {"=" * 60}"""

    LOGGER.info(banner)

    encoder_factory = get_encoder_for_model(args.model)
    encoder = encoder_factory()
    LOGGER.info("Encoder: %s", encoder.name)

    LOGGER.info("Loading dataset...")
    train_loader = create_dataloader(
        db_path=db_path,
        encoder=encoder,
        batch_size=training_cfg.batch_size,
        num_workers=settings.hardware.num_workers,
        include_value=(model_cfg.head in ["value", "dual"]),
        num_samples=args.num_samples,
    )
    LOGGER.info("Dataset size: %s positions", len(train_loader.dataset))  # type: ignore

    trainer = Trainer(
        model=model,
        device=device,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
        checkpoint_dir=str(checkpoint_dir),
    )

    if args.checkpoint:
        LOGGER.info("Resuming from: %s", args.checkpoint)
        trainer.load_checkpoint(args.checkpoint)
        LOGGER.info("Resuming from epoch %s", trainer.epoch)
    elif not args.no_auto_resume:
        if trainer.try_auto_resume():
            LOGGER.info("Auto-resumed from epoch %s", trainer.epoch)

    start_control_server(trainer, port=args.control_port)

    LOGGER.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=None,
        continuous=continuous,
    )

    LOGGER.info("Training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train chess models (supervised)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/train.py --model resnet --name baseline\n"
            "    → trains continuously, control API on :5050\n"
            "  python scripts/train.py --model resnet --epochs 20 --name baseline\n"
            "    → trains for 20 epochs then stops\n"
        ),
    )

    parser.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    parser.add_argument("--head", type=str, default=None, choices=["policy", "value", "dual"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name; organises outputs under runs/<name>/",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to .db, .parquet file, or directory of .parquet files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit training to N positions (default: use all)",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (default: from config)",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable automatic resume from latest checkpoint",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Train continuously until dataset is exhausted or stopped",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=5050,
        help="Port for the live hyperparameter control API (default: 5050)",
    )

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
