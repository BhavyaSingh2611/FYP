"""
Training loop for chess models.
"""
from pathlib import Path
from typing import Optional
import time
import requests

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from .losses import create_loss, PolicyLoss, ValueLoss, DualLoss
from ..models.base import ChessModel

NTFY_URL = "https://ntfy.lunex.page/FYP"


def _send_ntfy(title: str, message: str, priority: str = "default") -> None:
    try:
        requests.post(NTFY_URL,
            data=message.encode(encoding='utf-8'),
            headers={"Title": title, "Priority": priority})
    except Exception as e:
        print(f"Failed to send ntfy notification: {e}")


class Trainer:
    """
    Training loop for chess models.
    
    Features:
        - Support for all head types (policy, value, dual)
        - Learning rate scheduling
        - Checkpoint saving/loading
        - Mixed-precision (float16) training for MPS / CUDA
        - torch.compile support
    """
    
    def __init__(
        self,
        model: ChessModel,
        device: torch.device,
        head_type: str = "dual",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        use_soft_labels: bool = True,
        checkpoint_dir: Optional[str | Path] = None,
        mixed_precision: bool = True,
        compile_model: bool = True,
    ):
        self.device = device
        self.head_type = head_type

        self.use_amp = mixed_precision and device.type in ("cuda", "mps")
        if device.type == "cuda":
            self.amp_dtype = torch.float16
        elif device.type == "mps":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        self.model = model.to(device)
        is_gnn = type(model).__name__ in ("GCN", "GAT")
        if compile_model and not is_gnn and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        
        self.loss_fn = create_loss(
            head_type=head_type,
            policy_weight=policy_weight,
            value_weight=value_weight,
            use_soft_labels=use_soft_labels,
        )
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=self.use_amp and device.type == "cuda",
        )
        
        self.scheduler = None
        
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 100,
    ) -> dict:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0
        
        total = len(dataloader.dataset) // dataloader.batch_size if hasattr(dataloader.dataset, '__len__') else None
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=total)
        
        for batch_idx, batch in enumerate(pbar):
            inputs = self._prepare_inputs(batch['input'])
            policy_target = batch['policy_target'].to(self.device)
            value_target = batch.get('value_target')
            if value_target is not None:
                value_target = value_target.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model(inputs)
                if self.head_type == "dual":
                    loss_dict = self.loss_fn(output, policy_target, value_target)
                    loss = loss_dict['loss']
                    policy_loss_sum += loss_dict['policy_loss'].item()
                    value_loss_sum += loss_dict['value_loss'].item()
                elif self.head_type == "policy":
                    loss = self.loss_fn(output['policy'], policy_target)
                    policy_loss_sum += loss.item()
                else:
                    loss = self.loss_fn(output['value'], value_target)
                    value_loss_sum += loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute averages
        avg_loss = total_loss / num_batches
        metrics = {'loss': avg_loss}
        
        if self.head_type in ["dual", "policy"]:
            metrics['policy_loss'] = policy_loss_sum / num_batches
        if self.head_type in ["dual", "value"]:
            metrics['value_loss'] = value_loss_sum / num_batches
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = self._prepare_inputs(batch['input'])
            policy_target = batch['policy_target'].to(self.device)
            value_target = batch.get('value_target')
            if value_target is not None:
                value_target = value_target.to(self.device)
            
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model(inputs)
                if self.head_type == "dual":
                    loss_dict = self.loss_fn(output, policy_target, value_target)
                    loss = loss_dict['loss']
                    policy_loss_sum += loss_dict['policy_loss'].item()
                    value_loss_sum += loss_dict['value_loss'].item()
                elif self.head_type == "policy":
                    loss = self.loss_fn(output['policy'], policy_target)
                    policy_loss_sum += loss.item()
                else:
                    loss = self.loss_fn(output['value'], value_target)
                    value_loss_sum += loss.item()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracy for policy
            if 'policy' in output:
                pred = output['policy'].argmax(dim=-1)
                target_idx = policy_target.argmax(dim=-1)
                correct += (pred == target_idx).sum().item()
                total += pred.size(0)
        
        metrics = {'loss': total_loss / num_batches}
        
        if self.head_type in ["dual", "policy"]:
            metrics['policy_loss'] = policy_loss_sum / num_batches
            metrics['accuracy'] = correct / total if total > 0 else 0.0
        if self.head_type in ["dual", "value"]:
            metrics['value_loss'] = value_loss_sum / num_batches
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        scheduler_type: str = "cosine",
        save_best: bool = True,
        save_every: int = 10,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            epochs: Number of epochs.
            scheduler_type: LR scheduler type ("step", "cosine", "none").
            save_best: Save best model checkpoint.
            save_every: Save checkpoint every N epochs.
        
        Returns:
            Training history.
        """
        # Setup scheduler
        if scheduler_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        start_time = time.time()
        model_name = getattr(self.model, 'name', type(self.model).__name__)
        
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            
            if 'policy_loss' in train_metrics:
                print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
            if 'value_loss' in train_metrics:
                print(f"  Value Loss: {train_metrics['value_loss']:.4f}")
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                
                if 'accuracy' in val_metrics:
                    history['val_accuracy'].append(val_metrics['accuracy'])
                
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                if 'accuracy' in val_metrics:
                    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                
                if save_best and val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt')
                    print("  Saved best model!")
            
            if self.checkpoint_dir:
                self.save_checkpoint(f'epoch_{epoch}.pt')
                self.save_checkpoint('latest.pt')
            
            ntfy_lines = [f"Loss: {train_metrics['loss']:.4f}"]
            if 'policy_loss' in train_metrics:
                ntfy_lines.append(f"Policy: {train_metrics['policy_loss']:.4f}")
            if 'value_loss' in train_metrics:
                ntfy_lines.append(f"Value: {train_metrics['value_loss']:.4f}")
            _send_ntfy(
                title=f"{model_name} - Epoch {epoch}/{epochs}",
                message="\n".join(ntfy_lines),
            )
            
            print()
        
        total_time = time.time() - start_time
        print(f"Training complete in {total_time/60:.1f} minutes")
        
        if self.checkpoint_dir:
            self.save_checkpoint('final.pt')
            for ckpt in sorted(self.checkpoint_dir.glob("epoch_*.pt")):
                num = int(ckpt.stem.split("_")[1])
                if num % 5 != 0:
                    ckpt.unlink()
                    print(f"  Removed {ckpt.name}")
        
        _send_ntfy(
            title=f"{model_name} - Training Complete",
            message=f"Finished {epochs} epochs in {total_time/60:.1f} min\nFinal loss: {history['train_loss'][-1]:.4f}",
            priority="high",
        )
        
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'model_name': self.model.name,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _prepare_inputs(self, inputs):
        """Prepare inputs for model."""
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        elif isinstance(inputs, dict):
            return {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        else:
            return inputs
