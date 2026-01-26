# Training Documentation

## Overview

The training module (`src/training/`) provides everything needed to train chess models:

1. **Losses**: Policy, Value, and Dual loss functions
2. **Trainer**: Complete training loop with checkpointing

---

## Loss Functions

Location: `src/training/losses.py`

### PolicyLoss

Loss for move prediction. Supports both hard and soft labels.

#### Hard Labels (Cross-Entropy)

Used when the target is a single correct move.

```python
loss = -log(p_correct)
```

#### Soft Labels (KL Divergence)

Used when the target is a distribution from engine analysis.

```python
loss = sum(target * log(target / predicted))
```

#### Constructor

```python
PolicyLoss(
    use_soft_labels: bool = True,
    label_smoothing: float = 0.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_soft_labels` | bool | True | Use KL divergence for soft targets |
| `label_smoothing` | float | 0.0 | Label smoothing for cross-entropy |

#### Example

```python
from src.training.losses import PolicyLoss

loss_fn = PolicyLoss(use_soft_labels=True)

# Forward pass
policy_logits = model(input)['policy']  # (B, 4672)
target = batch['policy_target']          # (B, 4672) - soft distribution

loss = loss_fn(policy_logits, target)
```

---

### ValueLoss

Loss for board evaluation. Uses Mean Squared Error.

```python
loss = mean((predicted - target)^2)
```

#### Constructor

```python
ValueLoss()
```

No parameters.

#### Example

```python
from src.training.losses import ValueLoss

loss_fn = ValueLoss()

value = model(input)['value']        # (B, 1) in range [-1, 1]
target = batch['value_target']       # (B, 1)

loss = loss_fn(value, target)
```

---

### DualLoss

Combined policy and value loss for dual-headed models (AlphaZero-style).

```python
loss = policy_weight * policy_loss + value_weight * value_loss
```

#### Constructor

```python
DualLoss(
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    use_soft_labels: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_weight` | float | 1.0 | Weight for policy loss |
| `value_weight` | float | 1.0 | Weight for value loss |
| `use_soft_labels` | bool | True | Use soft labels for policy |

#### Example

```python
from src.training.losses import DualLoss

loss_fn = DualLoss(policy_weight=1.0, value_weight=1.0)

output = model(input)  # {'policy': (B, 4672), 'value': (B, 1)}
policy_target = batch['policy_target']
value_target = batch['value_target']

result = loss_fn(output, policy_target, value_target)
# result = {'loss': tensor, 'policy_loss': tensor, 'value_loss': tensor}
```

---

### create_loss Factory

```python
from src.training.losses import create_loss

loss_fn = create_loss(
    head_type="dual",
    policy_weight=1.0,
    value_weight=1.0,
    use_soft_labels=True,
)
```

| Head Type | Returns |
|-----------|---------|
| `policy` | PolicyLoss |
| `value` | ValueLoss |
| `dual` | DualLoss |

---

## Trainer

Location: `src/training/trainer.py`

Complete training loop with:

- Support for all head types
- Learning rate scheduling
- Checkpoint saving/loading
- Device optimization (MPS, CUDA)

### Constructor

```python
Trainer(
    model: ChessModel,
    device: torch.device,
    head_type: str = "dual",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    use_soft_labels: bool = True,
    checkpoint_dir: Optional[str | Path] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | ChessModel | - | Model to train |
| `device` | torch.device | - | Training device |
| `head_type` | str | "dual" | Head type (policy/value/dual) |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `weight_decay` | float | 0.0001 | L2 regularization |
| `policy_weight` | float | 1.0 | Policy loss weight |
| `value_weight` | float | 1.0 | Value loss weight |
| `use_soft_labels` | bool | True | Use soft policy labels |
| `checkpoint_dir` | str/Path | None | Checkpoint save directory |

### Methods

#### train_epoch

Trains for one epoch.

```python
metrics = trainer.train_epoch(
    dataloader=train_loader,
    epoch=1,
    log_interval=100,
)
# metrics = {'loss': 0.523, 'policy_loss': 0.312, 'value_loss': 0.211}
```

#### evaluate

Evaluates the model.

```python
metrics = trainer.evaluate(dataloader=val_loader)
# metrics = {'val_loss': 0.456, 'val_policy_loss': 0.289, ...}
```

#### train

Full training loop.

```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,      # Optional
    epochs=50,
    scheduler_type="cosine",    # "step", "cosine", "none"
    save_best=True,
    save_every=10,
)
```

Returns training history:
```python
{
    'train_loss': [0.8, 0.6, 0.5, ...],
    'val_loss': [0.7, 0.55, 0.45, ...],
    'train_policy_loss': [...],
    'train_value_loss': [...],
    'learning_rates': [...],
    'best_val_loss': 0.35,
    'best_epoch': 42,
}
```

#### save_checkpoint

Saves a training checkpoint.

```python
trainer.save_checkpoint("model_epoch_10.pt")
```

Checkpoint contents:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Current epoch
- `best_val_loss`: Best validation loss
- `model_name`: Model name string

#### load_checkpoint

Loads a training checkpoint.

```python
trainer.load_checkpoint("checkpoints/best_model.pt")
print(f"Resuming from epoch {trainer.epoch}")
```

---

## Learning Rate Schedulers

### Step Scheduler

Decays LR by gamma every step_size epochs.

```python
# lr = lr * gamma every step_size epochs
# Example: 0.001 -> 0.0001 -> 0.00001
```

### Cosine Annealing

Smoothly decays LR following a cosine curve.

```python
# lr = lr_min + (lr_max - lr_min) * (1 + cos(epoch/total_epochs * pi)) / 2
```

---

## Training Example

### Complete Training Script

```python
from src.config import load_config
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model
from src.data.dataset import create_dataloader
from src.training import Trainer

# Load configuration
config = load_config("config/default.yaml")
device = get_device()

# Create model
model = create_model(config.model)
print(f"Model: {model.name}, Parameters: {model.count_parameters():,}")

# Get encoder and create dataloader
encoder_factory = get_encoder_for_model(config.model.backbone)
encoder = encoder_factory()

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
    epochs=config.training.epochs,
    scheduler_type=config.training.lr_scheduler.type,
)

print(f"Best validation loss: {history['best_val_loss']}")
```

---

## Training Metrics

### Policy Metrics

| Metric | Description |
|--------|-------------|
| `policy_loss` | Cross-entropy or KL divergence loss |
| `top1_accuracy` | Accuracy of predicting the best move |
| `top5_accuracy` | Accuracy within top 5 predictions |

### Value Metrics

| Metric | Description |
|--------|-------------|
| `value_loss` | MSE between predicted and target value |
| `value_mae` | Mean absolute error |

### Combined Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Weighted sum of policy and value loss |

---

## Device Optimization

Location: `src/device.py`

### get_device

Automatically selects the best available device.

```python
from src.device import get_device

device = get_device(force_cpu=False, verbose=True)
# Device: MPS (Apple Silicon)
```

Priority order:
1. MPS (Apple Silicon)
2. CUDA (NVIDIA GPU)
3. CPU

### get_device_from_config

Gets device based on configuration.

```python
device = get_device_from_config(config.__dict__)
```

### optimize_for_device

Applies device-specific optimizations.

```python
from src.device import optimize_for_device

model = optimize_for_device(model, device)
```

Optimizations:
- CUDA: Enables TF32 and cuDNN benchmarking
- MPS: Moves model to MPS

---

## Checkpointing

### Checkpoint Directory Structure

```
checkpoints/
├── ResNet_10B_256C_epoch_10.pt
├── ResNet_10B_256C_epoch_20.pt
├── ResNet_10B_256C_best.pt
└── training_log.json
```

### Best Model Saving

When `save_best=True`, the trainer saves the model with the lowest validation loss as `{model_name}_best.pt`.

### Resuming Training

```python
# Load checkpoint
trainer.load_checkpoint("checkpoints/ResNet_10B_256C_epoch_10.pt")

# Continue training
history = trainer.train(
    train_loader=train_loader,
    epochs=50,  # Total epochs, will continue from checkpoint
)
```

---

## Troubleshooting

### Out of Memory

- Reduce batch size in config
- Use fewer data loading workers
- Switch to a smaller model

### Slow Training

- Increase number of workers
- Use MPS or CUDA if available
- Reduce model size for experimentation

### NaN Losses

- Check for zero-probability in soft labels
- Reduce learning rate
- Check for corrupted training data
