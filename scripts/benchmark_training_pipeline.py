#!/usr/bin/env python3
"""
Training pipeline benchmark — profiles every CPU and GPU stage of a
training iteration, per encoder/model pair.

Supports two modes:
  --num-workers 0  (default) : inline encoding with per-stage timing
  --num-workers N            : real DataLoader with N workers, measures
                               wall-clock data-loading vs GPU time

Usage:
  python scripts/benchmark_training_pipeline.py [--batch-size 256] [--iterations 50] [--warmup 5]
  python scripts/benchmark_training_pipeline.py --num-workers 4 --iterations 30
"""
import argparse
import multiprocessing
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.chess_env.encoders import CNNEncoder, TransformerEncoder, GNNEncoder
from src.chess_env.board_wrapper import UCI_MOVE_TO_INDEX, NUM_MOVES
from src.data.dataset import collate_fn, ChessDataset
from src.models.factory import create_backbone
from src.models.heads import create_head
from src.training.losses import create_loss


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    fen_parse_us: list[float] = field(default_factory=list)
    encode_us: list[float] = field(default_factory=list)
    policy_build_us: list[float] = field(default_factory=list)
    value_build_us: list[float] = field(default_factory=list)
    collate_us: list[float] = field(default_factory=list)
    to_device_us: list[float] = field(default_factory=list)
    forward_us: list[float] = field(default_factory=list)
    loss_us: list[float] = field(default_factory=list)
    backward_us: list[float] = field(default_factory=list)
    optimizer_us: list[float] = field(default_factory=list)


@dataclass
class WorkerTimings:
    dataload_us: list[float] = field(default_factory=list)
    to_device_us: list[float] = field(default_factory=list)
    forward_us: list[float] = field(default_factory=list)
    loss_us: list[float] = field(default_factory=list)
    backward_us: list[float] = field(default_factory=list)
    optimizer_us: list[float] = field(default_factory=list)


def build_position_pool(n: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(42)
    positions: list[tuple[str, str]] = []

    while len(positions) < n:
        board = chess.Board()
        num_moves = rng.integers(4, 80)
        for _ in range(num_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[rng.integers(0, len(legal))])
        legal = list(board.legal_moves)
        if legal:
            best = legal[rng.integers(0, len(legal))]
            positions.append((board.fen(), best.uci()))

    return positions[:n]


def make_temp_parquet(positions: list[tuple[str, str]], tmp_dir: Path) -> Path:
    fens = [p[0] for p in positions]
    moves = [p[1] for p in positions]
    rng = np.random.default_rng(99)
    cps = rng.integers(-400, 400, size=len(positions)).tolist()

    table = pa.table({
        "fen": fens,
        "best_move": moves,
        "cp": pa.array(cps, type=pa.int16()),
        "mate": pa.array([None] * len(positions), type=pa.int8()),
        "depth": pa.array([25] * len(positions), type=pa.uint8()),
        "value": [max(-1.0, min(1.0, c / 1000.0)) for c in cps],
    })
    path = tmp_dir / "bench.parquet"
    pq.write_table(table, path, row_group_size=10_000)
    return path


def time_ns():
    return time.perf_counter_ns()


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _prepare_inputs(inputs, device):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs


# ---------------------------------------------------------------------------
# Mode 1: Inline (num_workers=0) — per-stage breakdown
# ---------------------------------------------------------------------------

def benchmark_inline(
    name: str,
    encoder,
    backbone_type: str,
    positions: list[tuple[str, str]],
    batch_size: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> StageTimings:
    model = create_backbone(backbone_type)
    head = create_head("dual", model.get_backbone_output_dim(), 256)
    model.set_head(head)
    model = model.to(device)
    model.train()

    loss_fn = create_loss("dual", use_soft_labels=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    use_amp = device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp and device.type == "cuda")

    rng = np.random.default_rng(123)
    timings = StageTimings()

    for iteration in range(warmup + iterations):
        idx = rng.integers(0, len(positions), size=batch_size)
        batch_positions = [positions[i] for i in idx]

        examples = []
        batch_fen_us = 0.0
        batch_encode_us = 0.0
        batch_policy_us = 0.0
        batch_value_us = 0.0

        for fen_str, best_move in batch_positions:
            t0 = time_ns()
            board = chess.Board(fen_str)
            t1 = time_ns()

            encoded = encoder.encode(board)
            t2 = time_ns()

            policy = torch.zeros(NUM_MOVES)
            idx_move = UCI_MOVE_TO_INDEX.get(best_move, -1)
            if idx_move >= 0:
                policy[idx_move] = 1.0
            t3 = time_ns()

            cp_val = rng.integers(-400, 400)
            value = max(-1.0, min(1.0, cp_val / 1000.0))
            value_target = torch.tensor([value], dtype=torch.float32)
            t4 = time_ns()

            batch_fen_us += (t1 - t0) / 1000
            batch_encode_us += (t2 - t1) / 1000
            batch_policy_us += (t3 - t2) / 1000
            batch_value_us += (t4 - t3) / 1000

            examples.append({
                "input": encoded,
                "policy_target": policy,
                "value_target": value_target,
            })

        t0 = time_ns()
        batch = collate_fn(examples)
        t1 = time_ns()

        inputs = _prepare_inputs(batch["input"], device)
        policy_target = batch["policy_target"].to(device)
        value_target = batch["value_target"].to(device)
        sync_device(device)
        t2 = time_ns()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(inputs)
        sync_device(device)
        t3 = time_ns()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss_dict = loss_fn(output, policy_target, value_target)
            loss = loss_dict["loss"]
        sync_device(device)
        t4 = time_ns()

        scaler.scale(loss).backward()
        sync_device(device)
        t5 = time_ns()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        sync_device(device)
        t6 = time_ns()

        if iteration >= warmup:
            timings.fen_parse_us.append(batch_fen_us)
            timings.encode_us.append(batch_encode_us)
            timings.policy_build_us.append(batch_policy_us)
            timings.value_build_us.append(batch_value_us)
            timings.collate_us.append((t1 - t0) / 1000)
            timings.to_device_us.append((t2 - t1) / 1000)
            timings.forward_us.append((t3 - t2) / 1000)
            timings.loss_us.append((t4 - t3) / 1000)
            timings.backward_us.append((t5 - t4) / 1000)
            timings.optimizer_us.append((t6 - t5) / 1000)

    return timings


# ---------------------------------------------------------------------------
# Mode 2: DataLoader with workers — wall-clock data-loading vs GPU
# ---------------------------------------------------------------------------

def benchmark_with_workers(
    name: str,
    encoder,
    backbone_type: str,
    parquet_path: Path,
    batch_size: int,
    num_workers: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> WorkerTimings:
    model = create_backbone(backbone_type)
    head = create_head("dual", model.get_backbone_output_dim(), 256)
    model.set_head(head)
    model = model.to(device)
    model.train()

    loss_fn = create_loss("dual", use_soft_labels=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    use_amp = device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp and device.type == "cuda")

    dataset = ChessDataset(
        paths=[parquet_path],
        encoder=encoder,
        use_soft_labels=False,
        include_value=True,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    timings = WorkerTimings()
    count = 0

    for batch in loader:
        t0 = time_ns()
        # dataload time is measured as wall-clock between end of previous
        # iteration and when we receive the batch (t0 captured at start)

        inputs = _prepare_inputs(batch["input"], device)
        policy_target = batch["policy_target"].to(device)
        value_target = batch["value_target"].to(device)
        sync_device(device)
        t1 = time_ns()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(inputs)
        sync_device(device)
        t2 = time_ns()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss_dict = loss_fn(output, policy_target, value_target)
            loss = loss_dict["loss"]
        sync_device(device)
        t3 = time_ns()

        scaler.scale(loss).backward()
        sync_device(device)
        t4 = time_ns()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        sync_device(device)
        t5 = time_ns()

        count += 1
        if count > warmup:
            timings.to_device_us.append((t1 - t0) / 1000)
            timings.forward_us.append((t2 - t1) / 1000)
            timings.loss_us.append((t3 - t2) / 1000)
            timings.backward_us.append((t4 - t3) / 1000)
            timings.optimizer_us.append((t5 - t4) / 1000)

        if count >= warmup + iterations:
            break

    # Now measure dataload latency separately: how long the main thread
    # blocks waiting for the next batch from the DataLoader.
    loader2 = DataLoader(
        ChessDataset(
            paths=[parquet_path],
            encoder=encoder,
            use_soft_labels=False,
            include_value=True,
            shuffle=True,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    count = 0
    for batch in loader2:
        t_end = time_ns()
        count += 1
        if count > warmup and count <= warmup + iterations:
            timings.dataload_us.append((t_end - t_start) / 1000)
        if count >= warmup + iterations:
            break
        t_start = time_ns()

    return timings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_inline_report(all_results: dict[str, StageTimings], batch_size: int):
    stages = [
        ("fen_parse_us",    "1. FEN → Board",        "CPU"),
        ("encode_us",       "2. Board → Tensor",      "CPU"),
        ("policy_build_us", "3. Build policy target",  "CPU"),
        ("value_build_us",  "4. Build value target",   "CPU"),
        ("collate_us",      "5. Collate batch",        "CPU"),
        ("to_device_us",    "6. Transfer to device",   "CPU→GPU"),
        ("forward_us",      "7. Forward pass",         "GPU"),
        ("loss_us",         "8. Loss computation",     "GPU"),
        ("backward_us",     "9. Backward pass",        "GPU"),
        ("optimizer_us",    "10. Optimizer step",      "GPU"),
    ]

    for pipeline_name, timings in all_results.items():
        print(f"\n{'=' * 90}")
        print(f"  {pipeline_name}  (batch_size={batch_size}, num_workers=0)")
        print(f"{'=' * 90}")

        total_cpu_ms = 0.0
        total_gpu_ms = 0.0
        total_transfer_ms = 0.0
        rows = []

        for attr, label, where in stages:
            vals = np.array(getattr(timings, attr))
            mean_us = float(np.mean(vals))
            median_us = float(np.median(vals))
            p95_us = float(np.percentile(vals, 95))
            mean_ms = mean_us / 1000

            if where == "CPU":
                total_cpu_ms += mean_ms
            elif where == "GPU":
                total_gpu_ms += mean_ms
            else:
                total_transfer_ms += mean_ms

            rows.append((label, where, mean_us, median_us, p95_us, mean_ms))

        total_ms = total_cpu_ms + total_gpu_ms + total_transfer_ms

        hdr = f"  {'Stage':<26} {'Where':<8} {'Mean':>10} {'Median':>10} {'P95':>10} {'% Total':>9}"
        print(hdr)
        print("  " + "-" * 84)

        for label, where, mean_us, median_us, p95_us, mean_ms in rows:
            pct = 100 * mean_ms / total_ms if total_ms > 0 else 0
            if mean_us > 1000:
                mean_str = f"{mean_us / 1000:.2f}ms"
                median_str = f"{median_us / 1000:.2f}ms"
                p95_str = f"{p95_us / 1000:.2f}ms"
            else:
                mean_str = f"{mean_us:.1f}µs"
                median_str = f"{median_us:.1f}µs"
                p95_str = f"{p95_us:.1f}µs"
            print(f"  {label:<26} {where:<8} {mean_str:>10} {median_str:>10} {p95_str:>10} {pct:>8.1f}%")

        print("  " + "-" * 84)
        print(f"  {'CPU total':<26} {'':8} {total_cpu_ms:>9.2f}ms")
        print(f"  {'Transfer total':<26} {'':8} {total_transfer_ms:>9.2f}ms")
        print(f"  {'GPU total':<26} {'':8} {total_gpu_ms:>9.2f}ms")
        print(f"  {'ITERATION total':<26} {'':8} {total_ms:>9.2f}ms")
        print()

        cpu_pct = 100 * total_cpu_ms / total_ms if total_ms > 0 else 0
        gpu_pct = 100 * total_gpu_ms / total_ms if total_ms > 0 else 0
        throughput = batch_size / (total_ms / 1000) if total_ms > 0 else 0

        print(f"  CPU : {total_cpu_ms:.2f}ms ({cpu_pct:.1f}%)  |  GPU : {total_gpu_ms:.2f}ms ({gpu_pct:.1f}%)")
        print(f"  Throughput : {throughput:.0f} positions/sec")

        bottleneck = "CPU-bound (DataLoader)" if cpu_pct > gpu_pct else "GPU-bound (ideal)"
        print(f"  Bottleneck : {bottleneck}")


def print_worker_report(all_results: dict[str, WorkerTimings], batch_size: int, num_workers: int):
    stages = [
        ("dataload_us",   "1. DataLoader (CPU+collate)", "CPU"),
        ("to_device_us",  "2. Transfer to device",       "CPU→GPU"),
        ("forward_us",    "3. Forward pass",              "GPU"),
        ("loss_us",       "4. Loss computation",          "GPU"),
        ("backward_us",   "5. Backward pass",             "GPU"),
        ("optimizer_us",  "6. Optimizer step",            "GPU"),
    ]

    for pipeline_name, timings in all_results.items():
        print(f"\n{'=' * 90}")
        print(f"  {pipeline_name}  (batch_size={batch_size}, num_workers={num_workers})")
        print(f"{'=' * 90}")

        total_cpu_ms = 0.0
        total_gpu_ms = 0.0
        total_transfer_ms = 0.0
        rows = []

        for attr, label, where in stages:
            vals = np.array(getattr(timings, attr))
            if len(vals) == 0:
                rows.append((label, where, 0, 0, 0, 0))
                continue
            mean_us = float(np.mean(vals))
            median_us = float(np.median(vals))
            p95_us = float(np.percentile(vals, 95))
            mean_ms = mean_us / 1000

            if where == "CPU":
                total_cpu_ms += mean_ms
            elif where == "GPU":
                total_gpu_ms += mean_ms
            else:
                total_transfer_ms += mean_ms

            rows.append((label, where, mean_us, median_us, p95_us, mean_ms))

        total_ms = total_cpu_ms + total_gpu_ms + total_transfer_ms

        hdr = f"  {'Stage':<30} {'Where':<8} {'Mean':>10} {'Median':>10} {'P95':>10} {'% Total':>9}"
        print(hdr)
        print("  " + "-" * 88)

        for label, where, mean_us, median_us, p95_us, mean_ms in rows:
            pct = 100 * mean_ms / total_ms if total_ms > 0 else 0
            if mean_us > 1000:
                mean_str = f"{mean_us / 1000:.2f}ms"
                median_str = f"{median_us / 1000:.2f}ms"
                p95_str = f"{p95_us / 1000:.2f}ms"
            else:
                mean_str = f"{mean_us:.1f}µs"
                median_str = f"{median_us:.1f}µs"
                p95_str = f"{p95_us:.1f}µs"
            print(f"  {label:<30} {where:<8} {mean_str:>10} {median_str:>10} {p95_str:>10} {pct:>8.1f}%")

        print("  " + "-" * 88)

        gpu_total = total_gpu_ms + total_transfer_ms
        print(f"  {'DataLoader wait':<30} {'':8} {total_cpu_ms:>9.2f}ms")
        print(f"  {'GPU total (fwd+bwd+opt+xfer)':<30} {'':8} {gpu_total:>9.2f}ms")
        print(f"  {'ITERATION total':<30} {'':8} {total_ms:>9.2f}ms")
        print()

        throughput = batch_size / (total_ms / 1000) if total_ms > 0 else 0
        gpu_idle_pct = 100 * total_cpu_ms / total_ms if total_ms > 0 else 0

        print(f"  DataLoader : {total_cpu_ms:.2f}ms  |  GPU : {gpu_total:.2f}ms")
        print(f"  Throughput : {throughput:.0f} positions/sec")
        print(f"  GPU idle   : {gpu_idle_pct:.1f}% of iteration (waiting for data)")

        if total_cpu_ms > gpu_total:
            print(f"  Bottleneck : CPU-bound — try increasing --num-workers")
        else:
            print(f"  Bottleneck : GPU-bound (ideal)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark training pipeline stages")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = inline per-stage timing)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Mode: {'DataLoader with ' + str(args.num_workers) + ' workers' if args.num_workers > 0 else 'inline (per-stage breakdown)'}")

    pool_size = max(args.batch_size * (args.iterations + args.warmup), 5000)
    print(f"Building position pool ({pool_size} positions)...")
    positions = build_position_pool(pool_size)
    print(f"Pool ready. Running {args.iterations} iterations (warmup={args.warmup}).\n")

    pipelines = [
        ("ResNet + CNN encoder",           CNNEncoder(),                  "resnet"),
        ("SquareTransformer + Square enc", TransformerEncoder("square"),  "square_transformer"),
        ("PieceTransformer + Piece enc",   TransformerEncoder("piece"),   "piece_transformer"),
        ("GCN + GNN encoder",             GNNEncoder("hybrid"),          "gcn"),
    ]

    if args.num_workers == 0:
        all_results: dict[str, StageTimings] = {}
        for name, enc, backbone in pipelines:
            print(f"Benchmarking: {name} ...", flush=True)
            t = benchmark_inline(
                name, enc, backbone, positions,
                args.batch_size, args.iterations, args.warmup, device,
            )
            all_results[name] = t
            total_mean = (
                np.mean(t.fen_parse_us) + np.mean(t.encode_us) +
                np.mean(t.policy_build_us) + np.mean(t.value_build_us) +
                np.mean(t.collate_us) + np.mean(t.to_device_us) +
                np.mean(t.forward_us) + np.mean(t.loss_us) +
                np.mean(t.backward_us) + np.mean(t.optimizer_us)
            ) / 1000
            print(f"  → {total_mean:.1f}ms/iteration")

        print_inline_report(all_results, args.batch_size)

    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = make_temp_parquet(positions, Path(tmp_dir))
            print(f"Temp parquet: {parquet_path} ({len(positions):,} rows)\n")

            all_results_w: dict[str, WorkerTimings] = {}
            for name, enc, backbone in pipelines:
                print(f"Benchmarking: {name} ...", flush=True)
                t = benchmark_with_workers(
                    name, enc, backbone, parquet_path,
                    args.batch_size, args.num_workers,
                    args.iterations, args.warmup, device,
                )
                all_results_w[name] = t
                gpu_mean = (
                    np.mean(t.to_device_us) + np.mean(t.forward_us) +
                    np.mean(t.loss_us) + np.mean(t.backward_us) +
                    np.mean(t.optimizer_us)
                ) / 1000
                dl_mean = np.mean(t.dataload_us) / 1000 if t.dataload_us else 0
                print(f"  → GPU: {gpu_mean:.1f}ms  DataLoader: {dl_mean:.1f}ms")

            print_worker_report(all_results_w, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
