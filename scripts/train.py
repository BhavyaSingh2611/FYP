#!/usr/bin/env python3
"""
Script to train chess models.
"""
import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.device import get_device_from_config
from src.models.factory import create_model, get_encoder_for_model
from src.data.dataset import create_dataloader
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train chess models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model backbone (overrides config)"
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        choices=["policy", "value", "dual"],
        help="Head type (overrides config)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to training database (overrides config)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint output directory (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.model:
        config.model.backbone = args.model
    if args.head:
        config.model.head = args.head
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    db_path = args.database or config.paths.database
    checkpoint_dir = args.output_dir or config.paths.checkpoints
    
    # Get device - convert config object to dict format
    device_config = {"hardware": {"device": config.hardware.device}}
    device = get_device_from_config(device_config)
    
    print(f"Chess Model Trainer")
    print(f"=" * 40)
    print(f"Device: {device}")
    print(f"Model: {config.model.backbone} + {config.model.head} head")
    print(f"Database: {db_path}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(config.model)
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Get encoder
    encoder_factory = get_encoder_for_model(config.model.backbone)
    if callable(encoder_factory):
        encoder = encoder_factory()
    else:
        encoder = encoder_factory()
    
    print(f"Encoder: {encoder.name}")
    print()
    
    # Create data loader
    print("Loading dataset...")
    train_loader = create_dataloader(
        db_path=db_path,
        encoder=encoder,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        use_soft_labels=True,
        include_value=(config.model.head in ["value", "dual"]),
    )
    print(f"Dataset size: {len(train_loader.dataset)} positions")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        head_type=config.model.head,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        policy_weight=config.training.policy_loss_weight,
        value_weight=config.training.value_loss_weight,
        use_soft_labels=True,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        print(f"Resuming from epoch {trainer.epoch}")
        print()
    
    # Train
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=None,  # TODO: Add validation split
        epochs=config.training.epochs,
        scheduler_type=config.training.lr_scheduler.type,
        save_best=True,
        save_every=10,
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()
