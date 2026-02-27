#!/usr/bin/env python3
"""
Encoder benchmark — measures per-position encoding time and output size
for CNN, Transformer (Square & Piece), and GNN encoders.

Usage:
  python scripts/benchmark_encoders.py [--positions 1000] [--warmup 100] [--batch-sim 256]
"""
import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

from src.chess_env.encoders import CNNEncoder, TransformerEncoder, GNNEncoder


SAMPLE_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/5n2/3pN3/2PP4/8/PP2PPPP/RNBQKB1R b KQkq - 0 4",
    "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 0 8",
    "2r2rk1/pp1bppbp/2np1np1/q7/2BNP3/2N1BP2/PPPQ2PP/R4RK1 w - - 0 12",
    "r4rk1/1pp2ppp/p1np1q2/2b1p1B1/2B1P1b1/3P1N2/PPP2PPP/RN1Q1RK1 w - - 0 9",
    "8/5pk1/6p1/3P4/1p6/1P3PP1/5K2/8 w - - 0 40",
    "8/8/4k3/8/8/3K4/4P3/8 w - - 0 60",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def build_position_pool(n: int) -> list[chess.Board]:
    boards: list[chess.Board] = []
    rng = np.random.default_rng(42)

    for fen in SAMPLE_FENS:
        boards.append(chess.Board(fen))

    while len(boards) < n:
        board = chess.Board()
        num_moves = rng.integers(4, 80)
        for _ in range(num_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[rng.integers(0, len(legal))])
        boards.append(board)

    rng.shuffle(boards)
    return boards[:n]


def tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.nelement() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    return 0


def benchmark_encoder(name: str, encoder, boards: list[chess.Board], warmup: int):
    for b in boards[:warmup]:
        encoder.encode(b)

    n = len(boards)
    times = np.empty(n, dtype=np.float64)
    total_bytes = 0

    for i, board in enumerate(boards):
        t0 = time.perf_counter_ns()
        out = encoder.encode(board)
        t1 = time.perf_counter_ns()
        times[i] = (t1 - t0) / 1_000  # ns → µs
        total_bytes += tensor_bytes(out)

    avg_bytes = total_bytes / n
    return {
        "name": name,
        "mean_us": float(np.mean(times)),
        "median_us": float(np.median(times)),
        "p95_us": float(np.percentile(times, 95)),
        "p99_us": float(np.percentile(times, 99)),
        "min_us": float(np.min(times)),
        "max_us": float(np.max(times)),
        "std_us": float(np.std(times)),
        "avg_bytes": avg_bytes,
        "total_positions": n,
    }


def print_results(results: list[dict], batch_size: int):
    print("\n" + "=" * 80)
    print("ENCODER BENCHMARK RESULTS")
    print("=" * 80)

    hdr = f"{'Encoder':<28} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Bytes':>8}"
    print(hdr)
    print("-" * 80)
    for r in results:
        print(
            f"  {r['name']:<26} "
            f"{r['mean_us']:>7.1f}µs "
            f"{r['median_us']:>7.1f}µs "
            f"{r['p95_us']:>7.1f}µs "
            f"{r['p99_us']:>7.1f}µs "
            f"{r['avg_bytes']:>7.0f}B"
        )

    print()
    print(f"{'Encoder':<28} {'Batch':>10} {'Batches/s':>10} {'GPU starved?':>14}")
    print("-" * 80)
    gpu_forward_ms = 10.0
    for r in results:
        batch_ms = r["mean_us"] * batch_size / 1000
        batches_per_sec = 1000 / batch_ms if batch_ms > 0 else float("inf")
        workers_needed = batch_ms / gpu_forward_ms
        starved = "YES" if workers_needed > 8 else f"no ({workers_needed:.1f}w)"
        print(
            f"  {r['name']:<26} "
            f"{batch_ms:>8.1f}ms "
            f"{batches_per_sec:>9.1f} "
            f"{starved:>14}"
        )

    print()
    print("Notes:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Assumed GPU forward pass: {gpu_forward_ms:.0f}ms per batch")
    print(f"  - 'GPU starved?' = workers needed to match GPU throughput (>8 = bottleneck)")
    print(f"  - Positions benchmarked: {results[0]['total_positions']}")

    print()
    print("Storage projection (100M positions):")
    print(f"  {'Encoder':<28} {'FEN (~60B)':>12} {'Pre-encoded':>14} {'Ratio':>8}")
    print("  " + "-" * 62)
    fen_total = 100_000_000 * 60
    for r in results:
        pre_total = 100_000_000 * r["avg_bytes"]
        ratio = pre_total / fen_total
        print(
            f"  {r['name']:<28} "
            f"{fen_total / 1e9:>10.1f} GB "
            f"{pre_total / 1e9:>12.1f} GB "
            f"{ratio:>7.0f}×"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark chess state encoders")
    parser.add_argument("--positions", type=int, default=2000, help="Number of positions to benchmark")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations (not timed)")
    parser.add_argument("--batch-sim", type=int, default=256, help="Batch size for throughput projection")
    args = parser.parse_args()

    print(f"Building pool of {args.positions} random positions...")
    boards = build_position_pool(args.positions)
    print(f"Pool ready ({len(boards)} boards, warmup={args.warmup})\n")

    encoders = [
        ("CNN (18×8×8)", CNNEncoder()),
        ("Transformer-Square (64 tok)", TransformerEncoder("square")),
        ("Transformer-Piece (≤32 tok)", TransformerEncoder("piece")),
        ("GNN-Hybrid (64 nodes)", GNNEncoder("hybrid")),
    ]

    results = []
    for name, enc in encoders:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        r = benchmark_encoder(name, enc, boards, args.warmup)
        print(f"{r['mean_us']:.1f}µs/pos")
        results.append(r)

    print_results(results, args.batch_sim)


if __name__ == "__main__":
    main()
