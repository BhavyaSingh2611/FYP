"""
Check for duplicate FEN positions across the dataset.

Strategy (memory-friendly for 844M rows):
  1. HyperLogLog pass — estimates unique count in ~256KB of memory.
  2. Random sample pass — draws ~2M FENs, counts exact dupes in sample
     to characterise the duplication pattern.

Parallelised across files with multiprocessing (M1 Pro, 8 workers).
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xxhash
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCH_SIZE = 200_000
NUM_WORKERS = 8

HLL_P = 18
HLL_M = 1 << HLL_P
HLL_MASK = HLL_M - 1

RESERVOIR_PER_FILE = 200_000


def _nlz(val: int, bits: int) -> int:
    if val == 0:
        return bits
    return bits - val.bit_length()


@dataclass
class FileResult:
    total_rows: int = 0
    registers: np.ndarray = field(default_factory=lambda: np.zeros(HLL_M, dtype=np.uint8))
    sample: list[str] = field(default_factory=list)


def process_file(args: tuple[Path, int]) -> FileResult:
    fpath, worker_idx = args
    res = FileResult()
    regs = res.registers
    rng = np.random.default_rng(hash(fpath.name) & 0xFFFFFFFF)

    pf = pq.ParquetFile(fpath)
    num_batches = math.ceil(pf.metadata.num_rows / BATCH_SIZE)

    pbar = tqdm(
        pf.iter_batches(batch_size=BATCH_SIZE, columns=["fen"]),
        total=num_batches,
        desc=fpath.name,
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        position=worker_idx,
        leave=False,
    )

    bits = 64 - HLL_P
    for batch in pbar:
        fens = batch.column("fen").to_pylist()
        for fen in fens:
            if fen is None:
                continue
            res.total_rows += 1
            h = xxhash.xxh64_intdigest(fen)

            idx = h & HLL_MASK
            w = h >> HLL_P
            rho = _nlz(w, bits) + 1
            if rho > regs[idx]:
                regs[idx] = rho

            if len(res.sample) < RESERVOIR_PER_FILE:
                res.sample.append(fen)
            else:
                j = rng.integers(0, res.total_rows)
                if j < RESERVOIR_PER_FILE:
                    res.sample[int(j)] = fen

    return res


def hll_count(registers: np.ndarray) -> float:
    m = len(registers)
    alpha = 0.7213 / (1 + 1.079 / m)
    raw = alpha * m * m / np.sum(np.power(2.0, -registers.astype(np.float64)))
    if raw <= 2.5 * m:
        zeros = np.sum(registers == 0)
        if zeros > 0:
            return m * math.log(m / zeros)
    return raw


if __name__ == "__main__":
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Using {NUM_WORKERS} worker processes (M1 Pro)")
    print(f"HyperLogLog registers: {HLL_M:,} (p={HLL_P}, ~{HLL_M // 1024} KB)")
    print(f"Reservoir sample: {RESERVOIR_PER_FILE:,} per file\n")

    mp.set_start_method("fork", force=True)
    jobs = [(fpath, i % NUM_WORKERS) for i, fpath in enumerate(parquet_files)]
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_file, jobs)
    print("\n" * NUM_WORKERS)

    total_rows = sum(r.total_rows for r in results)
    merged_regs = np.zeros(HLL_M, dtype=np.uint8)
    for r in results:
        np.maximum(merged_regs, r.registers, out=merged_regs)

    rng = np.random.default_rng(42)
    all_samples: list[str] = []
    for r in results:
        all_samples.extend(r.sample)
    target = 2_000_000
    if len(all_samples) > target:
        idx = rng.choice(len(all_samples), target, replace=False)
        all_samples = [all_samples[i] for i in idx]

    estimated_unique = hll_count(merged_regs)
    estimated_dup_rate = 1.0 - estimated_unique / total_rows
    hll_error_pct = 1.04 / math.sqrt(HLL_M) * 100

    print(f"{'=' * 65}")
    print("RESULTS")
    print(f"{'=' * 65}")
    print(f"  Total FEN rows      : {total_rows:,}")
    print(f"  Estimated unique    : {estimated_unique:,.0f}")
    print(f"  Estimated duplicates: {total_rows - estimated_unique:,.0f}")
    print(f"  Duplicate rate      : {estimated_dup_rate * 100:.2f}%")
    print(f"  HLL std error       : ±{hll_error_pct:.2f}%")

    print(f"\n  --- Sample analysis ({len(all_samples):,} FENs) ---")
    fen_counts = Counter(all_samples)
    unique_in_sample = len(fen_counts)
    duped_in_sample = sum(1 for c in fen_counts.values() if c > 1)

    print(f"  Unique in sample    : {unique_in_sample:,}")
    print(f"  Duplicated FENs     : {duped_in_sample:,}")
    print(f"  Sample dup rate     : {duped_in_sample / unique_in_sample * 100:.2f}%")
    print(f"\n  Top 10 most repeated FENs in sample:")
    for fen, count in fen_counts.most_common(10):
        print(f"    {count:>4}x | {fen}")
