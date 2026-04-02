"""
Training loop for chess models.
"""

import contextlib
import json
import logging
import math
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import requests
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ModelConfig, TrainingConfig
from ..models.base import ChessModel
from .losses import create_loss

NTFY_URL = "https://ntfy.lunex.page/FYP"
LOGGER = logging.getLogger(__name__)

# Architecture sets used for device-specific optimisations
_CNN_MODELS = ("ResNet", "ConvNet")
_GNN_MODELS = ("GCN", "GAT")


def _send_ntfy(title: str, message: str, priority: str = "default") -> None:
    try:
        requests.post(
            NTFY_URL,
            data=message.encode(encoding="utf-8"),
            headers={"Title": title, "Priority": priority},
        )
    except Exception as e:
        LOGGER.error(f"Failed to send ntfy notification: {e}")


class Trainer:
    """
    Training loop for chess models.

    Features:
        - Support for all head types (policy, value, dual)
        - Learning rate scheduling (driven by config)
        - Checkpoint saving/loading
        - Mixed-precision (float16) training for MPS / CUDA
        - torch.compile support
    """

    def __init__(
        self,
        model: ChessModel,
        device: torch.device,
        training_cfg: TrainingConfig,
        model_cfg: ModelConfig,
        checkpoint_dir: str | Path | None = None,
    ):
        self.training_cfg = training_cfg
        self.head_type = model_cfg.head
        self.device = device

        # ---- Mixed precision ----

        self.use_amp = device.type in ("cuda", "mps")

        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif device.type in ("cuda", "mps"):
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # ---- Model setup ----

        self.model = model.to(device)
        arch = type(model).__name__
        self._use_channels_last = arch in _CNN_MODELS and device.type == "cuda"

        if self._use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]

        if arch not in _GNN_MODELS and hasattr(torch, "compile"):
            with contextlib.suppress(Exception):
                self.model = cast(ChessModel, torch.compile(self.model))

        # ---- Loss & optimiser ----

        self.loss_fn = create_loss(
            head_type=self.head_type,
            policy_weight=training_cfg.policy_loss_weight,
            value_weight=training_cfg.value_loss_weight,
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )

        # Grad scaler: only for float16 AMP on CUDA
        use_scaler = self.use_amp and device.type == "cuda" and self.amp_dtype != torch.bfloat16
        self.scaler = torch.amp.grad_scaler.GradScaler(device=device.type, enabled=use_scaler)

        # ---- State ----

        self.scheduler: StepLR | CosineAnnealingLR | None = None
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self._should_stop = False

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.stats_history: list[dict] = []
        self._stats_path = (self.checkpoint_dir / "training_stats.jsonl") if self.checkpoint_dir else None

    # ------------------------------------------------------------------
    # Stats persistence
    # ------------------------------------------------------------------

    def _save_stats_entry(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict | None,
        epoch_time: float,
    ) -> None:
        """Append a stats entry for this epoch to history and JSONL file."""
        entry: dict = {
            "epoch": epoch,
            "global_step": self.global_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch_time_seconds": epoch_time,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        for k, v in train_metrics.items():
            entry[f"train_{k}"] = v

        if val_metrics:
            for k, v in val_metrics.items():
                entry[f"val_{k}"] = v

        cfg = self.training_cfg
        entry["learning_rate"] = cfg.learning_rate
        entry["policy_loss_weight"] = cfg.policy_loss_weight
        entry["value_loss_weight"] = cfg.value_loss_weight
        entry["grad_clip_max_norm"] = cfg.grad_clip_max_norm
        entry["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps

        self.stats_history.append(entry)

        if self._stats_path:
            with open(self._stats_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, inputs: torch.Tensor | dict) -> torch.Tensor | dict:
        """Move inputs to device, applying channels-last for CNN tensors."""
        if isinstance(inputs, torch.Tensor):
            t = inputs.to(self.device, non_blocking=True)
            return t.to(memory_format=torch.channels_last) if self._use_channels_last and t.ndim == 4 else t

        if isinstance(inputs, dict):
            return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}

        return inputs

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor | dict, torch.Tensor, torch.Tensor | None]:
        """Extract inputs and targets from a batch dict, moving to device."""
        inputs = self._prepare_inputs(batch["input"])
        policy_target = batch["policy_target"].to(self.device, non_blocking=True)

        value_target = batch.get("value_target")
        if value_target is not None:
            value_target = value_target.to(self.device, non_blocking=True)

        return inputs, policy_target, value_target

    def _compute_loss(
        self,
        output: dict,
        policy_target: torch.Tensor,
        value_target: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute loss and return (loss, policy_loss, value_loss) based on head type."""
        if self.head_type == "dual":
            loss_dict = self.loss_fn(output, policy_target, value_target)
            return loss_dict["loss"], loss_dict["policy_loss"].item(), loss_dict["value_loss"].item()

        if self.head_type == "policy":
            loss = self.loss_fn(output["policy"], policy_target)
            return loss, loss.item(), 0.0

        # value-only
        loss = self.loss_fn(output["value"], value_target)
        return loss, 0.0, loss.item()

    def _build_metrics(
        self,
        total_loss: float,
        policy_sum: float,
        value_sum: float,
        n: int,
        correct: int = 0,
        total: int = 0,
    ) -> dict[str, float]:
        """Build a metrics dict from running sums."""
        metrics: dict[str, float] = {"loss": total_loss / n}

        if self.head_type in ("dual", "policy"):
            metrics["policy_loss"] = policy_sum / n
            if total > 0:
                metrics["accuracy"] = correct / total

        if self.head_type in ("dual", "value"):
            metrics["value_loss"] = value_sum / n

        return metrics

    # ------------------------------------------------------------------
    # Single-epoch training (pure compute, no logging)
    # ------------------------------------------------------------------

    def _run_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Run one training epoch. Returns metrics dict."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = policy_sum = value_sum = 0.0
        num_batches = 0

        accum = self.training_cfg.gradient_accumulation_steps
        max_norm = self.training_cfg.grad_clip_max_norm

        num_total = (
            len(dataloader.dataset) // dataloader.batch_size  # type: ignore
            if hasattr(dataloader.dataset, "__len__")
            else None
        )
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=num_total)

        for batch_idx, batch in enumerate(pbar):
            inputs, policy_target, value_target = self._unpack_batch(batch)

            # Forward
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model(inputs)
                loss, p_loss, v_loss = self._compute_loss(output, policy_target, value_target)
                loss = loss / accum

            # Backward
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == num_total:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Accumulate
            total_loss += loss.item() * accum
            policy_sum += p_loss
            value_sum += v_loss
            num_batches += 1
            self.global_step += 1

            if batch_idx % 100 == 0:
                pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        if self.scheduler is not None:
            self.scheduler.step()

        return self._build_metrics(total_loss, policy_sum, value_sum, num_batches)

    # ------------------------------------------------------------------
    # Evaluation (pure compute, no logging)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate the model. Returns metrics dict."""
        self.model.eval()

        total_loss = policy_sum = value_sum = 0.0
        correct = total = num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, policy_target, value_target = self._unpack_batch(batch)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model(inputs)
                loss, p_loss, v_loss = self._compute_loss(output, policy_target, value_target)

            total_loss += loss.item()
            policy_sum += p_loss
            value_sum += v_loss
            num_batches += 1

            # Policy accuracy
            if "policy" in output:
                pred = output["policy"].argmax(dim=-1)
                correct += (pred == policy_target.argmax(dim=-1)).sum().item()
                total += pred.size(0)

        return self._build_metrics(total_loss, policy_sum, value_sum, num_batches, correct, total)

    # ------------------------------------------------------------------
    # Full training loop (orchestration + logging)
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        continuous: bool = False,
    ) -> dict:
        """
        Full training loop.  All hyper-parameters come from ``self.training_cfg``.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            continuous: If True, train indefinitely (ignoring ``epochs``)
                until the dataset is exhausted or a stop signal is received.

        Returns:
            Training history.
        """
        cfg = self.training_cfg
        sched_cfg = cfg.lr_scheduler
        epochs = cfg.epochs
        model_name = getattr(self.model, "name", type(self.model).__name__)

        if continuous:
            LOGGER.info("Continuous training mode — will run until data is exhausted or stopped")

        # ---- Signal handling ----

        orig_sigterm = signal.getsignal(signal.SIGTERM)
        orig_sigint = signal.getsignal(signal.SIGINT)

        sigint_times: list[float] = []
        sigint_window_seconds = 5.0
        sigint_force_exit_count = 3

        def _handle_stop(signum, frame):
            if signum == signal.SIGINT:
                now = time.monotonic()
                sigint_times[:] = [t for t in sigint_times if now - t <= sigint_window_seconds]
                sigint_times.append(now)

                if len(sigint_times) >= sigint_force_exit_count:
                    LOGGER.warning(
                        "Received Ctrl+C %d times within %.0f seconds; exiting immediately.",
                        len(sigint_times),
                        sigint_window_seconds,
                    )
                    raise KeyboardInterrupt

                remaining = max(0, sigint_force_exit_count - len(sigint_times))
                LOGGER.info(
                    "Received Ctrl+C, will stop after current epoch. "
                    "Press Ctrl+C %d more time(s) within %.0f seconds to exit immediately.",
                    remaining,
                    sigint_window_seconds,
                )
                self._should_stop = True
                return

            LOGGER.info(f"Received signal {signum}, will stop after current epoch")
            self._should_stop = True

        signal.signal(signal.SIGTERM, _handle_stop)
        signal.signal(signal.SIGINT, _handle_stop)

        # ---- Scheduler setup (skipped in continuous mode) ----

        if not continuous:
            if sched_cfg.type == "step":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=sched_cfg.step_size,
                    gamma=sched_cfg.gamma,
                )
            elif sched_cfg.type == "cosine":
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
            else:
                self.scheduler = None
        else:
            self.scheduler = None

        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        start_time = time.time()

        # ---- Epoch loop ----

        start_epoch = self.epoch + 1
        epoch = start_epoch - 1
        forced_immediate_stop = False

        def _epoch_iter():
            """Yield epoch numbers: finite range or infinite counter."""
            if continuous:
                e = start_epoch
                while True:
                    yield e
                    e += 1
            else:
                yield from range(start_epoch, epochs + 1)

        try:
            for epoch in _epoch_iter():
                self.epoch = epoch
                display_epochs = "∞" if continuous else str(epochs)

                epoch_start = time.time()
                train_metrics = self._run_epoch(train_loader, epoch)
                epoch_time = time.time() - epoch_start

                # In continuous mode, an epoch with 0 batches means
                # the dataset iterator is exhausted.
                if math.isnan(train_metrics["loss"]):
                    LOGGER.info("Dataset exhausted (NaN loss) — stopping")
                    break

                history["train_loss"].append(train_metrics["loss"])

                # Validation
                val_metrics = self.evaluate(val_loader) if val_loader else None

                if val_metrics:
                    history["val_loss"].append(val_metrics["loss"])

                    if "accuracy" in val_metrics:
                        history["val_accuracy"].append(val_metrics["accuracy"])

                # Checkpointing
                tracked_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
                is_best = tracked_loss < self.best_loss
                if is_best:
                    self.best_loss = tracked_loss
                    self.save_checkpoint("best.pt")

                self.save_checkpoint("latest.pt")
                if epoch % cfg.save_every == 0:
                    self.save_checkpoint(f"epoch_{epoch}.pt")

                # Logging
                self._log_epoch(epoch, display_epochs, train_metrics, val_metrics, model_name, is_best)

                self._save_stats_entry(epoch, train_metrics, val_metrics, epoch_time)

                # Graceful stop on signal
                if self._should_stop:
                    self.save_checkpoint("latest.pt")
                    _send_ntfy(
                        title=f"{model_name} - Training Interrupted",
                        message=f"Stopped after epoch {epoch}/{display_epochs}",
                        priority="high",
                    )
                    break
        except KeyboardInterrupt:
            forced_immediate_stop = True
            self.save_checkpoint("latest.pt")
            _send_ntfy(
                title=f"{model_name} - Training Force Stopped",
                message=(
                    "Exited immediately due to repeated Ctrl+C "
                    f"(threshold: {sigint_force_exit_count} within {sigint_window_seconds:.0f}s)"
                ),
                priority="high",
            )
        finally:
            signal.signal(signal.SIGTERM, orig_sigterm)
            signal.signal(signal.SIGINT, orig_sigint)

        # ---- Finish ----

        total_time = time.time() - start_time

        if forced_immediate_stop:
            LOGGER.info(f"Training force-stopped in {total_time / 60:.1f} minutes")
            return history

        LOGGER.info(f"Training complete in {total_time / 60:.1f} minutes")

        final_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
        display_epochs = "∞" if continuous else str(epochs)

        _send_ntfy(
            title=f"{model_name} - Training Complete",
            message=(f"Finished {epoch} epochs in {total_time / 60:.1f} minutes\nFinal loss: {final_loss:.4f}"),
            priority="high",
        )

        return history

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch: int,
        epochs: str,
        train_metrics: dict,
        val_metrics: dict | None,
        model_name: str,
        is_best: bool = False,
    ) -> None:
        """Log epoch summary and send push notification."""

        # Console / LOGGER
        lines = [
            f"Epoch {epoch}/{epochs}",
            f"  Train Loss: {train_metrics['loss']:.4f}",
        ]

        if "policy_loss" in train_metrics:
            lines.append(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")

        if "value_loss" in train_metrics:
            lines.append(f"  Value Loss: {train_metrics['value_loss']:.4f}")

        if val_metrics:
            lines.append(f"  Val Loss: {val_metrics['loss']:.4f}")

            if "accuracy" in val_metrics:
                lines.append(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

        if is_best:
            lines.append("  ★ Saved best model")

        LOGGER.info("\n".join(lines))

        # Push notification
        ntfy_lines = [f"Loss: {train_metrics['loss']:.4f}"]

        if "policy_loss" in train_metrics:
            ntfy_lines.append(f"Policy: {train_metrics['policy_loss']:.4f}")

        if "value_loss" in train_metrics:
            ntfy_lines.append(f"Value: {train_metrics['value_loss']:.4f}")

        _send_ntfy(
            title=f"{model_name} - Epoch {epoch}/{epochs}",
            message="\n".join(ntfy_lines),
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "model_name": self.model.name,
            "training_cfg": self.training_cfg.dict(),
            "stats_history_len": len(self.stats_history),
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)  # nosec: checkpoint may contain non-tensor objects

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))

        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if self._stats_path and self._stats_path.exists():
            self.stats_history = []
            with open(self._stats_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.stats_history.append(json.loads(line))

    def try_auto_resume(self) -> bool:
        """Resume from ``latest.pt`` if it exists. Returns True if resumed."""
        if not self.checkpoint_dir:
            return False

        latest = self.checkpoint_dir / "latest.pt"
        if not latest.exists():
            return False

        LOGGER.info(f"Resuming from {latest}")
        self.load_checkpoint(latest)
        return True
