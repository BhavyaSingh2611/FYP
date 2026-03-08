"""
Dataset analysis script – processes all Parquet files in parallel using
multiprocessing (one file per worker), then merges results and produces
summary statistics + plots.

Parquet schema: f (FEN), b (best move UCI), v (value -1 to +1).
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "data_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000
NUM_WORKERS = 8
RESERVOIR_CAP_PER_FILE = 200_000

VALUE_HIST_EDGES = np.linspace(-1.0, 1.0, 201)


@dataclass
class FileResult:
    total_rows: int = 0
    val_sum: float = 0.0
    val_sq_sum: float = 0.0
    val_count: int = 0
    val_min: float = float("inf")
    val_max: float = float("-inf")
    val_hist: np.ndarray = field(default_factory=lambda: np.zeros(200, dtype=np.int64))
    white_to_move: int = 0
    black_to_move: int = 0


def process_file(args: tuple[Path, int]) -> FileResult:
    fpath, worker_idx = args
    res = FileResult(
        val_hist=np.zeros(len(VALUE_HIST_EDGES) - 1, dtype=np.int64),
    )

    pf = pq.ParquetFile(fpath)
    num_batches = math.ceil(pf.metadata.num_rows / BATCH_SIZE)

    pbar = tqdm(
        pf.iter_batches(batch_size=BATCH_SIZE),
        total=num_batches,
        desc=fpath.name,
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        position=worker_idx,
        leave=False,
    )

    for batch in pbar:
        n = batch.num_rows
        res.total_rows += n

        v_arr = batch.column("v").to_numpy(zero_copy_only=False).astype(np.float64)
        mask_v = ~np.isnan(v_arr)
        v_valid = v_arr[mask_v]
        if len(v_valid):
            res.val_sum += v_valid.sum()
            res.val_sq_sum += (v_valid**2).sum()
            res.val_count += len(v_valid)
            res.val_min = min(res.val_min, float(v_valid.min()))
            res.val_max = max(res.val_max, float(v_valid.max()))
            res.val_hist += np.histogram(v_valid, bins=VALUE_HIST_EDGES)[0]

        fen_arr = batch.column("f").to_pylist()
        for fen in fen_arr:
            if fen and " w " in fen:
                res.white_to_move += 1
            elif fen:
                res.black_to_move += 1

    return res


def merge_results(results: list[FileResult]) -> FileResult:
    merged = FileResult(
        val_hist=np.zeros(len(VALUE_HIST_EDGES) - 1, dtype=np.int64),
    )

    for r in results:
        merged.total_rows += r.total_rows
        merged.val_sum += r.val_sum
        merged.val_sq_sum += r.val_sq_sum
        merged.val_count += r.val_count
        merged.val_min = min(merged.val_min, r.val_min)
        merged.val_max = max(merged.val_max, r.val_max)
        merged.val_hist += r.val_hist
        merged.white_to_move += r.white_to_move
        merged.black_to_move += r.black_to_move

    return merged


if __name__ == "__main__":
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {DATA_DIR}")
    print(f"Using {NUM_WORKERS} worker processes\n")

    mp.set_start_method("fork", force=True)
    jobs = [(fpath, i % NUM_WORKERS) for i, fpath in enumerate(parquet_files)]
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_file, jobs)
    print("\n" * NUM_WORKERS)

    r = merge_results(results)
    total_rows = r.total_rows

    print(f"\nTotal rows processed: {total_rows:,}\n")

    print("=" * 65)
    print("VALUE DISTRIBUTION (v)")
    print("=" * 65)
    val_mean = r.val_sum / r.val_count if r.val_count else 0
    val_std = ((r.val_sq_sum / r.val_count) - val_mean**2) ** 0.5 if r.val_count else 0
    print(f"  Count (non-null) : {r.val_count:,}")
    print(f"  Mean             : {val_mean:+.4f}")
    print(f"  Std              : {val_std:.4f}")
    print(f"  Min              : {r.val_min:+.4f}")
    print(f"  Max              : {r.val_max:+.4f}")

    print()
    print("=" * 65)
    print("ACTIVE COLOUR")
    print("=" * 65)
    print(
        f"  White to move : {r.white_to_move:,}  ({100 * r.white_to_move / total_rows:.2f}%)"
    )
    print(
        f"  Black to move : {r.black_to_move:,}  ({100 * r.black_to_move / total_rows:.2f}%)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chess Dataset Analysis", fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0]
    centres = (VALUE_HIST_EDGES[:-1] + VALUE_HIST_EDGES[1:]) / 2
    ax.bar(centres, r.val_hist, width=0.01, color="steelblue", edgecolor="none")
    ax.set_xlabel("Value (v)")
    ax.set_ylabel("Count")
    ax.set_title("Value Distribution (-1 to +1)")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    ax = axes[1]
    ax.pie(
        [r.white_to_move, r.black_to_move],
        labels=["White to move", "Black to move"],
        autopct="%1.1f%%",
        colors=["#f0d9b5", "#b58863"],
        textprops={"fontsize": 12},
    )
    ax.set_title("Active Colour Split")

    plt.tight_layout()
    out_path = OUT_DIR / "dataset_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")

    report_path = OUT_DIR / "dataset_analysis.txt"
    with open(report_path, "w") as f:
        f.write(f"Total rows: {total_rows:,}\n\n")
        f.write(
            f"Value — mean: {val_mean:+.4f}, std: {val_std:.4f}, "
            f"min: {r.val_min:+.4f}, max: {r.val_max:+.4f}, non-null: {r.val_count:,}\n"
        )
        f.write(
            f"White to move: {r.white_to_move:,}, Black to move: {r.black_to_move:,}\n"
        )
    print(f"Text report saved to {report_path}")
