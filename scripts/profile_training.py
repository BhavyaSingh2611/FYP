#!/usr/bin/env python3
"""
Profile a short training run to identify bottlenecks.

Runs a configurable number of steps with per-phase timing
(data loading, encoding, forward, backward, optimizer) and
reports where time is spent.

Usage:
    python scripts/profile_training.py --model resnet --steps 50
    python scripts/profile_training.py --model gat --steps 100 --batch-size 128
"""

import argparse
import logging
import time

import torch

from src.config import settings
from src.data.dataset import create_dataloader
from src.device import get_device
from src.models.factory import create_model, get_encoder_for_model
from src.training.losses import create_loss
from src.training.profiler import StepProfiler, benchmark

LOGGER = logging.getLogger(__name__)

ALL_MODELS = [
    "convnet", "resnet", "square_transformer",
    "piece_transformer", "gcn", "gat",
]


@benchmark
def warmup_gpu(model, device, sample_batch):
    """Run a few dummy forward passes to warm up GPU kernels / torch.compile."""
    model.train()
    for _ in range(3):
        with torch.no_grad():
            model(sample_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_profile(args):
    device = get_device()
    model_cfg = settings.model.model_copy(update={"head": "dual"})
    training_cfg = settings.training

    batch_size = args.batch_size or training_cfg.batch_size
    db_path = args.database or settings.paths.database

    # ---- Model ----
    model = create_model(args.model, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("Model: %s  |  params: %s  |  device: %s", args.model, f"{n_params:,}", device)

    loss_fn = create_loss(
        head_type="dual",
        policy_weight=training_cfg.policy_loss_weight,
        value_weight=training_cfg.value_loss_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    use_amp = device.type in ("cuda", "mps")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device.type in ("cuda", "mps"):
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    use_scaler = use_amp and device.type == "cuda" and amp_dtype != torch.bfloat16
    scaler = torch.amp.grad_scaler.GradScaler(device=device.type, enabled=use_scaler)

    # ---- Data ----
    encoder = get_encoder_for_model(args.model)()
    loader = create_dataloader(
        db_path=db_path,
        encoder=encoder,
        batch_size=batch_size,
        num_workers=settings.hardware.num_workers,
        include_value=True,
        num_samples=args.steps * batch_size * 2,  # load enough data
    )

    # ---- Warmup ----
    data_iter = iter(loader)
    first_batch = next(data_iter)

    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in x.items()}
        return x

    sample_input = _to_device(first_batch["input"])
    warmup_gpu(model, device, sample_input)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- Profiled steps ----
    sp = StepProfiler()
    model.train()

    LOGGER.info("Running %d profiled steps (batch_size=%d) ...", args.steps, batch_size)
    wall_start = time.perf_counter()

    step = 0
    for batch in loader:
        if step >= args.steps:
            break

        # Phase 1: data transfer
        with sp.phase("data_to_device"):
            inputs = _to_device(batch["input"])
            policy_target = batch["policy_target"].to(device, non_blocking=True)
            value_target = batch["value_target"].to(device, non_blocking=True)

        # Phase 2: forward
        with sp.phase("forward"):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(inputs)

        # Phase 3: loss
        with sp.phase("loss"):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss_dict = loss_fn(output, policy_target, value_target)
                loss = loss_dict["loss"]

        # Phase 4: backward
        with sp.phase("backward"):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

        # Phase 5: optimizer step
        with sp.phase("optim_step"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        sp.step_done()
        step += 1

    wall_elapsed = time.perf_counter() - wall_start
    throughput = (step * batch_size) / wall_elapsed if wall_elapsed > 0 else 0

    LOGGER.info("Wall time: %.2fs  |  throughput: %.0f samples/s", wall_elapsed, throughput)
    sp.report()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Profile training bottlenecks")
    parser.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps to profile")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--database", type=str, default=None, help="Path to parquet data")

    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
