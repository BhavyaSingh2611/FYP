#!/usr/bin/env python3
"""
Self-play training script.

Trains a model by:
1. Loading existing checkpoint
2. Generating self-play games using MCTS
3. Training on the generated data
4. Optionally repeating for multiple iterations
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm

from src.config import load_config
from src.models.factory import create_model, get_encoder_for_model
from src.training.self_play import SelfPlayGenerator, games_to_tensors
from src.device import get_device


def train_on_self_play(
    model,
    data: dict,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
):
    """
    Train model on self-play data.
    
    Args:
        model: Model to train.
        data: Dict with input tensors, 'policies', 'values'.
        device: Training device.
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
    
    Returns:
        Training history.
    """
    model.train()
    
    is_dict_input = data.get('is_dict_input', False)
    n_samples = len(data['policies'])
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    history = {'loss': [], 'policy_loss': [], 'value_loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        n_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        
        for start in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            policies = data['policies'][batch_idx].to(device)
            values = data['values'][batch_idx].to(device).unsqueeze(1)
            
            # Create model input based on type
            if is_dict_input:
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
            
            # Policy loss: cross-entropy with soft targets
            policy_logits = output['policy']
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(policies * log_probs).sum(dim=-1).mean()
            
            # Value loss: MSE
            value_pred = output['value']
            value_loss = F.mse_loss(value_pred, values)
            
            # Combined loss
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




def main():
    parser = argparse.ArgumentParser(description="Self-play training")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["convnet", "resnet", "square_transformer", "piece_transformer"],
                       help="Model to train")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path (default: training_results/<model>/final.pt)")
    parser.add_argument("--games", type=int, default=20,
                       help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=100,
                       help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs per iteration")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of self-play + train iterations")
    parser.add_argument("--output-dir", type=str, default="self_play_results",
                       help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just test without training")
    
    args = parser.parse_args()
    
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SELF-PLAY TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games per iteration: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Training epochs: {args.epochs}")
    print(f"Iterations: {args.iterations}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    config = load_config("config/default.yaml")
    config.model.backbone = args.model
    config.model.head = "dual"
    
    model = create_model(config.model)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = f"training_results/{args.model}/final.pt"
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Get encoder
    encoder_factory = get_encoder_for_model(args.model)
    encoder = encoder_factory() if callable(encoder_factory) else encoder_factory()
    
    if args.dry_run:
        print("\n[DRY RUN] Testing self-play generation...")
        generator = SelfPlayGenerator(
            model=model,
            encoder=encoder,
            device=device,
            num_simulations=10,  # Quick test
        )
        game = generator.generate_game(max_moves=10)
        print(f"Generated test game: {len(game.moves)} moves, result: {game.result}")
        print("Dry run successful!")
        return
    
    # Self-play training loop
    all_history = []
    
    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")
        
        # Generate self-play games
        print(f"\nGenerating {args.games} self-play games...")
        generator = SelfPlayGenerator(
            model=model,
            encoder=encoder,
            device=device,
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
        
        # Convert to tensors
        print("\nConverting to training data...")
        data = games_to_tensors(games, encoder, model_type=args.model)
        n_examples = len(data.get('inputs', data.get('tokens', [])))
        print(f"Training data: {n_examples} examples")
        
        # Train
        print(f"\nTraining for {args.epochs} epochs...")
        history = train_on_self_play(
            model=model,
            data=data,
            device=device,
            epochs=args.epochs,
        )
        
        all_history.append({
            'iteration': iteration + 1,
            'games': args.games,
            'positions': total_positions,
            'results': results,
            'history': history,
        })
        
        # Save checkpoint
        checkpoint_name = f"{args.model}_selfplay_iter{iteration + 1}.pt"
        checkpoint_path = output_dir / checkpoint_name
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model.name,
            'iteration': iteration + 1,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / f"{args.model}_selfplay_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.name,
        'total_iterations': args.iterations,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save training log
    log_path = output_dir / f"{args.model}_selfplay_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'model': args.model,
            'config': {
                'games_per_iter': args.games,
                'simulations': args.simulations,
                'epochs': args.epochs,
                'iterations': args.iterations,
            },
            'history': all_history,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Training log saved: {log_path}")
    
    print("\n" + "=" * 60)
    print("SELF-PLAY TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTo benchmark the improved model:")
    print(f"  python scripts/benchmark_detailed.py \\")
    print(f"    --checkpoint-dir {output_dir} --games 4")


if __name__ == "__main__":
    main()
