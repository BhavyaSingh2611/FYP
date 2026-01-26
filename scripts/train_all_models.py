#!/usr/bin/env python3
"""
Train all models and generate comparative metrics & visualizations.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np

# All available model backbones
ALL_MODELS = [
    "convnet",
    "resnet",
    "square_transformer",
    "piece_transformer",
    "gcn",
    "gat",
]


def train_model(model_name: str, epochs: int, batch_size: int, database: str, output_dir: Path) -> dict:
    """
    Train a single model and return metrics.
    """
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Run training
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model", model_name,
        "--head", "dual",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--database", database,
        "--output-dir", str(model_output_dir),
    ]
    
    start_time = time.time()
    
    # Capture output
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    elapsed_time = time.time() - start_time
    
    # Parse training output to extract metrics
    output = result.stdout
    stderr = result.stderr
    
    print(output)
    if result.returncode != 0:
        print(f"STDERR: {stderr}")
    
    # Extract metrics from output
    metrics = parse_training_output(output, elapsed_time)
    metrics['model'] = model_name
    metrics['success'] = result.returncode == 0
    
    return metrics


def parse_training_output(output: str, elapsed_time: float) -> dict:
    """
    Parse training output to extract metrics.
    """
    metrics = {
        'train_losses': [],
        'policy_losses': [],
        'value_losses': [],
        'epochs_completed': 0,
        'final_loss': None,
        'final_policy_loss': None,
        'final_value_loss': None,
        'parameters': None,
        'elapsed_time': elapsed_time,
    }
    
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Parse parameter count
        if 'Parameters:' in line:
            try:
                params_str = line.split('Parameters:')[1].strip().replace(',', '')
                metrics['parameters'] = int(params_str)
            except:
                pass
        
        # Parse epoch metrics
        if 'Train Loss:' in line:
            try:
                loss = float(line.split('Train Loss:')[1].strip())
                metrics['train_losses'].append(loss)
                metrics['epochs_completed'] = len(metrics['train_losses'])
            except:
                pass
        
        if 'Policy Loss:' in line:
            try:
                loss = float(line.split('Policy Loss:')[1].strip())
                metrics['policy_losses'].append(loss)
            except:
                pass
        
        if 'Value Loss:' in line:
            try:
                loss = float(line.split('Value Loss:')[1].strip())
                metrics['value_losses'].append(loss)
            except:
                pass
    
    # Set final metrics
    if metrics['train_losses']:
        metrics['final_loss'] = metrics['train_losses'][-1]
    if metrics['policy_losses']:
        metrics['final_policy_loss'] = metrics['policy_losses'][-1]
    if metrics['value_losses']:
        metrics['final_value_loss'] = metrics['value_losses'][-1]
    
    return metrics


def generate_visualizations(all_metrics: list, output_dir: Path):
    """
    Generate comparative visualizations for all models.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful runs
    successful = [m for m in all_metrics if m.get('success', False)]
    
    if not successful:
        print("No successful training runs to visualize!")
        return
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful)))
    
    # 1. Training Loss Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(successful):
        if m['train_losses']:
            ax.plot(range(1, len(m['train_losses']) + 1), m['train_losses'], 
                   label=m['model'], color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(figures_dir / 'training_loss_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'training_loss_curves.png'}")
    
    # 2. Policy Loss Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(successful):
        if m['policy_losses']:
            ax.plot(range(1, len(m['policy_losses']) + 1), m['policy_losses'], 
                   label=m['model'], color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Policy Loss', fontsize=12)
    ax.set_title('Policy Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(figures_dir / 'policy_loss_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'policy_loss_curves.png'}")
    
    # 3. Value Loss Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(successful):
        if m['value_losses']:
            ax.plot(range(1, len(m['value_losses']) + 1), m['value_losses'], 
                   label=m['model'], color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value Loss', fontsize=12)
    ax.set_title('Value Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(figures_dir / 'value_loss_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'value_loss_curves.png'}")
    
    # 4. Final Loss Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = [m['model'] for m in successful]
    x = np.arange(len(models))
    
    # Total Loss
    final_losses = [m['final_loss'] or 0 for m in successful]
    bars = axes[0].bar(x, final_losses, color=colors[:len(successful)])
    axes[0].set_xlabel('Model', fontsize=10)
    axes[0].set_ylabel('Final Total Loss', fontsize=10)
    axes[0].set_title('Final Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Policy Loss
    policy_losses = [m['final_policy_loss'] or 0 for m in successful]
    axes[1].bar(x, policy_losses, color=colors[:len(successful)])
    axes[1].set_xlabel('Model', fontsize=10)
    axes[1].set_ylabel('Final Policy Loss', fontsize=10)
    axes[1].set_title('Final Policy Loss', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Value Loss
    value_losses = [m['final_value_loss'] or 0 for m in successful]
    axes[2].bar(x, value_losses, color=colors[:len(successful)])
    axes[2].set_xlabel('Model', fontsize=10)
    axes[2].set_ylabel('Final Value Loss', fontsize=10)
    axes[2].set_title('Final Value Loss', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'final_losses_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'final_losses_comparison.png'}")
    
    # 5. Model Parameters Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    params = [m['parameters'] or 0 for m in successful]
    bars = ax.bar(models, params, color=colors[:len(successful)])
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Parameters (millions)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    # Format y-axis in millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_parameters.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'model_parameters.png'}")
    
    # 6. Training Time Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    times = [m['elapsed_time'] / 60 for m in successful]  # Convert to minutes
    bars = ax.bar(models, times, color=colors[:len(successful)])
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'training_time.png', dpi=150)
    plt.close()
    print(f"Saved: {figures_dir / 'training_time.png'}")
    
    # 7. Summary Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training curves (top left)
    ax = axes[0, 0]
    for i, m in enumerate(successful):
        if m['train_losses']:
            ax.plot(range(1, len(m['train_losses']) + 1), m['train_losses'], 
                   label=m['model'], color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Final losses (top right)
    ax = axes[0, 1]
    width = 0.25
    x = np.arange(len(models))
    ax.bar(x - width, final_losses, width, label='Total', color='steelblue')
    ax.bar(x, policy_losses, width, label='Policy', color='coral')
    ax.bar(x + width, value_losses, width, label='Value', color='seagreen')
    ax.set_xlabel('Model')
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Loss Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Parameters (bottom left)
    ax = axes[1, 0]
    ax.bar(models, params, color=colors[:len(successful)])
    ax.set_xlabel('Model')
    ax.set_ylabel('Parameters')
    ax.set_title('Model Size', fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Training time (bottom right)
    ax = axes[1, 1]
    ax.bar(models, times, color=colors[:len(successful)])
    ax.set_xlabel('Model')
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Duration', fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Training Results Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / 'training_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'training_dashboard.png'}")


def generate_report(all_metrics: list, output_dir: Path):
    """
    Generate a markdown report with training results.
    """
    report_path = output_dir / "training_report.md"
    
    successful = [m for m in all_metrics if m.get('success', False)]
    failed = [m for m in all_metrics if not m.get('success', False)]
    
    with open(report_path, 'w') as f:
        f.write("# Chess Model Training Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Models Trained:** {len(successful)}\n")
        f.write(f"- **Failed:** {len(failed)}\n\n")
        
        if failed:
            f.write("### Failed Models\n\n")
            for m in failed:
                f.write(f"- {m['model']}\n")
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        
        # Create comparison table
        f.write("| Model | Parameters | Final Loss | Policy Loss | Value Loss | Time (min) |\n")
        f.write("|-------|------------|------------|-------------|------------|------------|\n")
        
        for m in sorted(successful, key=lambda x: x.get('final_loss', float('inf'))):
            params_str = f"{m['parameters']:,}" if m['parameters'] else "N/A"
            loss_str = f"{m['final_loss']:.4f}" if m['final_loss'] else "N/A"
            policy_str = f"{m['final_policy_loss']:.4f}" if m['final_policy_loss'] else "N/A"
            value_str = f"{m['final_value_loss']:.4f}" if m['final_value_loss'] else "N/A"
            time_str = f"{m['elapsed_time']/60:.1f}"
            
            f.write(f"| {m['model']} | {params_str} | {loss_str} | {policy_str} | {value_str} | {time_str} |\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("![Training Dashboard](figures/training_dashboard.png)\n\n")
        f.write("### Individual Charts\n\n")
        f.write("- [Training Loss Curves](figures/training_loss_curves.png)\n")
        f.write("- [Policy Loss Curves](figures/policy_loss_curves.png)\n")
        f.write("- [Value Loss Curves](figures/value_loss_curves.png)\n")
        f.write("- [Final Losses Comparison](figures/final_losses_comparison.png)\n")
        f.write("- [Model Parameters](figures/model_parameters.png)\n")
        f.write("- [Training Time](figures/training_time.png)\n\n")
        
        f.write("## Recommendations\n\n")
        
        if successful:
            # Find best model
            best_model = min(successful, key=lambda x: x.get('final_loss', float('inf')))
            f.write(f"**Best Performing Model:** `{best_model['model']}`\n")
            f.write(f"- Final Loss: {best_model['final_loss']:.4f}\n\n")
            
            # Check if more training is needed
            f.write("### Training Progress Assessment\n\n")
            for m in successful:
                if m['train_losses'] and len(m['train_losses']) >= 3:
                    # Check if loss is still decreasing
                    last_losses = m['train_losses'][-3:]
                    avg_improvement = (last_losses[0] - last_losses[-1]) / last_losses[0] * 100
                    
                    if avg_improvement > 5:
                        f.write(f"- **{m['model']}:** Loss still decreasing ({avg_improvement:.1f}% in last 3 epochs). Consider more training.\n")
                    elif avg_improvement > 1:
                        f.write(f"- **{m['model']}:** Moderate improvement ({avg_improvement:.1f}%). May benefit from more epochs.\n")
                    else:
                        f.write(f"- **{m['model']}:** Converged ({avg_improvement:.1f}% change). Training appears sufficient.\n")
    
    print(f"\nReport saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train all chess models")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs per model (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/chess_dataset.db",
        help="Path to training database"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_results",
        help="Output directory for checkpoints and figures"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Specific models to train (default: all)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CHESS MODEL TRAINING - ALL ARCHITECTURES")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Database: {args.database}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    all_metrics = []
    
    for model_name in args.models:
        try:
            metrics = train_model(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                database=args.database,
                output_dir=output_dir,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            all_metrics.append({
                'model': model_name,
                'success': False,
                'error': str(e),
            })
    
    # Save raw metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert non-serializable values
        serializable = []
        for m in all_metrics:
            s = {}
            for k, v in m.items():
                if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                    s[k] = v
            serializable.append(s)
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(all_metrics, output_dir)
    
    # Generate report
    print("\nGenerating training report...")
    generate_report(all_metrics, output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"- Checkpoints: {output_dir}/<model_name>/")
    print(f"- Visualizations: {output_dir}/figures/")
    print(f"- Report: {output_dir}/training_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
