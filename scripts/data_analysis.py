"""
Dataset analysis script – processes all Parquet files in parallel using
multiprocessing (one file per worker), then merges results and produces
summary statistics + plots.

Tuned for Apple M1 Pro (8 performance cores).
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "data_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000
NUM_WORKERS = 8
RESERVOIR_CAP_PER_FILE = 200_000


@dataclass
class FileResult:
    total_rows: int = 0
    cp_sum: float = 0.0
    cp_sq_sum: float = 0.0
    cp_count: int = 0
    cp_min: float = float("inf")
    cp_max: float = float("-inf")
    cp_hist: np.ndarray = field(default_factory=lambda: np.zeros(240, dtype=np.int64))
    cp_abs_hist: np.ndarray = field(default_factory=lambda: np.zeros(121, dtype=np.int64))
    depth_counts: Counter = field(default_factory=Counter)
    kn_sum: float = 0.0
    kn_sq_sum: float = 0.0
    kn_count: int = 0
    kn_min: float = float("inf")
    kn_max: float = float("-inf")
    kn_reservoir: list[float] = field(default_factory=list)
    mate_count: int = 0
    mate_dist: Counter = field(default_factory=Counter)
    pv_len_sum: int = 0
    pv_len_counts: Counter = field(default_factory=Counter)
    white_to_move: int = 0
    black_to_move: int = 0


CP_HIST_EDGES = np.arange(-3000, 3025, 25)
CP_ABS_HIST_EDGES = np.arange(0, 3025, 25)


def process_file(args: tuple[Path, int]) -> FileResult:
    fpath, worker_idx = args
    res = FileResult(
        cp_hist=np.zeros(len(CP_HIST_EDGES) - 1, dtype=np.int64),
        cp_abs_hist=np.zeros(len(CP_ABS_HIST_EDGES) - 1, dtype=np.int64),
    )
    rng = np.random.default_rng(hash(fpath.name) & 0xFFFFFFFF)

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

        # --- centipawn ---
        cp_arr = batch.column("cp").to_numpy(zero_copy_only=False)
        mask_cp = ~np.isnan(cp_arr.astype(float))
        cp_valid = cp_arr[mask_cp].astype(np.float64)
        if len(cp_valid):
            res.cp_sum += cp_valid.sum()
            res.cp_sq_sum += (cp_valid ** 2).sum()
            res.cp_count += len(cp_valid)
            res.cp_min = min(res.cp_min, float(cp_valid.min()))
            res.cp_max = max(res.cp_max, float(cp_valid.max()))
            res.cp_hist += np.histogram(cp_valid, bins=CP_HIST_EDGES)[0]
            res.cp_abs_hist += np.histogram(np.abs(cp_valid), bins=CP_ABS_HIST_EDGES)[0]

        # --- depth ---
        depth_arr = batch.column("depth").to_numpy(zero_copy_only=False)
        mask_d = ~np.isnan(depth_arr.astype(float))
        d_valid = depth_arr[mask_d].astype(int)
        for d, c in zip(*np.unique(d_valid, return_counts=True)):
            res.depth_counts[int(d)] += int(c)

        # --- knodes ---
        kn_arr = batch.column("knodes").to_numpy(zero_copy_only=False)
        mask_kn = ~np.isnan(kn_arr.astype(float))
        kn_valid = kn_arr[mask_kn].astype(np.float64)
        if len(kn_valid):
            res.kn_sum += kn_valid.sum()
            res.kn_sq_sum += (kn_valid ** 2).sum()
            res.kn_count += len(kn_valid)
            res.kn_min = min(res.kn_min, float(kn_valid.min()))
            res.kn_max = max(res.kn_max, float(kn_valid.max()))
            if len(res.kn_reservoir) < RESERVOIR_CAP_PER_FILE:
                take = min(len(kn_valid), RESERVOIR_CAP_PER_FILE - len(res.kn_reservoir))
                res.kn_reservoir.extend(kn_valid[:take].tolist())
            else:
                for v in kn_valid:
                    j = rng.integers(0, res.total_rows)
                    if j < RESERVOIR_CAP_PER_FILE:
                        res.kn_reservoir[int(j)] = float(v)

        # --- mate ---
        mate_arr = batch.column("mate").to_pylist()
        for m in mate_arr:
            if m is not None:
                res.mate_count += 1
                res.mate_dist[m] += 1

        # --- PV length ---
        line_arr = batch.column("line").to_pylist()
        for line in line_arr:
            if line:
                pv_len = len(line.split())
                res.pv_len_sum += pv_len
                res.pv_len_counts[pv_len] += 1

        # --- active colour ---
        fen_arr = batch.column("fen").to_pylist()
        for fen in fen_arr:
            if fen and " w " in fen:
                res.white_to_move += 1
            elif fen:
                res.black_to_move += 1

    return res


def merge_results(results: list[FileResult]) -> FileResult:
    merged = FileResult(
        cp_hist=np.zeros(len(CP_HIST_EDGES) - 1, dtype=np.int64),
        cp_abs_hist=np.zeros(len(CP_ABS_HIST_EDGES) - 1, dtype=np.int64),
    )
    all_reservoirs: list[float] = []

    for r in results:
        merged.total_rows += r.total_rows
        merged.cp_sum += r.cp_sum
        merged.cp_sq_sum += r.cp_sq_sum
        merged.cp_count += r.cp_count
        merged.cp_min = min(merged.cp_min, r.cp_min)
        merged.cp_max = max(merged.cp_max, r.cp_max)
        merged.cp_hist += r.cp_hist
        merged.cp_abs_hist += r.cp_abs_hist
        merged.depth_counts += r.depth_counts
        merged.kn_sum += r.kn_sum
        merged.kn_sq_sum += r.kn_sq_sum
        merged.kn_count += r.kn_count
        merged.kn_min = min(merged.kn_min, r.kn_min)
        merged.kn_max = max(merged.kn_max, r.kn_max)
        all_reservoirs.extend(r.kn_reservoir)
        merged.mate_count += r.mate_count
        merged.mate_dist += r.mate_dist
        merged.pv_len_sum += r.pv_len_sum
        merged.pv_len_counts += r.pv_len_counts
        merged.white_to_move += r.white_to_move
        merged.black_to_move += r.black_to_move

    rng = np.random.default_rng(42)
    if len(all_reservoirs) > 2_000_000:
        idx = rng.choice(len(all_reservoirs), 2_000_000, replace=False)
        merged.kn_reservoir = [all_reservoirs[i] for i in idx]
    else:
        merged.kn_reservoir = all_reservoirs

    return merged


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {DATA_DIR}")
    print(f"Using {NUM_WORKERS} worker processes (M1 Pro)\n")

    mp.set_start_method("fork", force=True)
    jobs = [(fpath, i % NUM_WORKERS) for i, fpath in enumerate(parquet_files)]
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_file, jobs)
    print("\n" * NUM_WORKERS)

    r = merge_results(results)
    total_rows = r.total_rows

    print(f"\nTotal rows processed: {total_rows:,}\n")

    # ── print summary ────────────────────────────────────────────────────────
    print("=" * 65)
    print("CENTIPAWN EVALUATION (cp)")
    print("=" * 65)
    cp_mean = r.cp_sum / r.cp_count if r.cp_count else 0
    cp_std = ((r.cp_sq_sum / r.cp_count) - cp_mean ** 2) ** 0.5 if r.cp_count else 0
    print(f"  Count (non-null) : {r.cp_count:,}")
    print(f"  Mean             : {cp_mean:+.2f}")
    print(f"  Std              : {cp_std:.2f}")
    print(f"  Min              : {r.cp_min}")
    print(f"  Max              : {r.cp_max}")
    null_cp = total_rows - r.cp_count
    print(f"  Null (mate pos.) : {null_cp:,}  ({100 * null_cp / total_rows:.2f}%)")

    print()
    print("=" * 65)
    print("DEPTH DISTRIBUTION")
    print("=" * 65)
    for d in sorted(r.depth_counts):
        pct = 100 * r.depth_counts[d] / total_rows
        print(f"  depth {d:>3}: {r.depth_counts[d]:>12,}  ({pct:5.2f}%)")

    print()
    print("=" * 65)
    print("KNODES (kilo-nodes searched)")
    print("=" * 65)
    kn_mean = r.kn_sum / r.kn_count if r.kn_count else 0
    kn_std = ((r.kn_sq_sum / r.kn_count) - kn_mean ** 2) ** 0.5 if r.kn_count else 0
    kn_res = np.array(r.kn_reservoir)
    print(f"  Count (non-null) : {r.kn_count:,}")
    print(f"  Mean             : {kn_mean:,.0f}")
    print(f"  Std              : {kn_std:,.0f}")
    print(f"  Min              : {r.kn_min:,.0f}")
    print(f"  Max              : {r.kn_max:,.0f}")
    if len(kn_res):
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"  P{p:<3}             : {np.percentile(kn_res, p):,.0f}")

    print()
    print("=" * 65)
    print("MATE EVALUATION")
    print("=" * 65)
    print(f"  Positions with mate : {r.mate_count:,}  ({100 * r.mate_count / total_rows:.2f}%)")
    if r.mate_dist:
        sorted_mates = sorted(r.mate_dist.items(), key=lambda x: -x[1])[:20]
        print("  Top 20 mate-in-N values:")
        for m, c in sorted_mates:
            print(f"    mate {m:>4}: {c:>10,}")

    print()
    print("=" * 65)
    print("PRINCIPAL VARIATION LENGTH")
    print("=" * 65)
    pv_total = sum(r.pv_len_counts.values())
    pv_mean = r.pv_len_sum / pv_total if pv_total else 0
    print(f"  Mean PV length : {pv_mean:.2f} moves")
    pv_sorted = sorted(r.pv_len_counts.items())
    print(f"  Min PV length  : {pv_sorted[0][0] if pv_sorted else 'N/A'}")
    print(f"  Max PV length  : {pv_sorted[-1][0] if pv_sorted else 'N/A'}")

    print()
    print("=" * 65)
    print("ACTIVE COLOUR")
    print("=" * 65)
    print(f"  White to move : {r.white_to_move:,}  ({100 * r.white_to_move / total_rows:.2f}%)")
    print(f"  Black to move : {r.black_to_move:,}  ({100 * r.black_to_move / total_rows:.2f}%)")

    # ── plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Chess Dataset Analysis", fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    centres = (CP_HIST_EDGES[:-1] + CP_HIST_EDGES[1:]) / 2
    ax.bar(centres, r.cp_hist, width=25, color="steelblue", edgecolor="none")
    ax.set_xlim(-1500, 1500)
    ax.set_xlabel("Centipawn evaluation")
    ax.set_ylabel("Count")
    ax.set_title("Centipawn Distribution (clipped ±1500)")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    ax = axes[0, 1]
    abs_centres = (CP_ABS_HIST_EDGES[:-1] + CP_ABS_HIST_EDGES[1:]) / 2
    ax.bar(abs_centres, r.cp_abs_hist, width=25, color="coral", edgecolor="none")
    ax.set_xlim(0, 1500)
    ax.set_xlabel("|Centipawn|")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Centipawn Distribution (clipped 1500)")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    ax = axes[1, 0]
    depths_sorted = sorted(r.depth_counts.items())
    ds, dc = zip(*depths_sorted) if depths_sorted else ([], [])
    ax.bar(ds, dc, color="mediumseagreen", edgecolor="none")
    ax.set_xlabel("Stockfish Depth")
    ax.set_ylabel("Count")
    ax.set_title("Depth Distribution")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    ax = axes[1, 1]
    if len(kn_res):
        kn_log = np.log10(kn_res[kn_res > 0] + 1)
        ax.hist(kn_log, bins=100, color="mediumpurple", edgecolor="none")
        ax.set_xlabel("log₁₀(knodes)")
        ax.set_ylabel("Count (reservoir sample)")
        ax.set_title("Kilo-Nodes Searched (log scale)")

    ax = axes[2, 0]
    pv_keys = sorted(r.pv_len_counts.keys())
    pv_vals = [r.pv_len_counts[k] for k in pv_keys]
    ax.bar(pv_keys, pv_vals, color="goldenrod", edgecolor="none")
    ax.set_xlim(0, max(pv_keys[:60]) if len(pv_keys) > 60 else None)
    ax.set_xlabel("PV Length (# moves)")
    ax.set_ylabel("Count")
    ax.set_title("Principal Variation Length")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    ax = axes[2, 1]
    ax.pie(
        [r.white_to_move, r.black_to_move],
        labels=["White to move", "Black to move"],
        autopct="%1.1f%%",
        colors=["#f0d9b5", "#b58863"],
        textprops={"fontsize": 12},
    )
    ax.set_title("Active Colour Split")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "dataset_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")

    report_path = OUT_DIR / "dataset_analysis.txt"
    with open(report_path, "w") as f:
        f.write(f"Total rows: {total_rows:,}\n\n")
        f.write(f"CP  — mean: {cp_mean:+.2f}, std: {cp_std:.2f}, "
                f"min: {r.cp_min}, max: {r.cp_max}, non-null: {r.cp_count:,}\n")
        f.write(f"Depth distribution: { {d: r.depth_counts[d] for d in sorted(r.depth_counts)} }\n")
        f.write(f"Knodes — mean: {kn_mean:,.0f}, std: {kn_std:,.0f}, "
                f"min: {r.kn_min:,.0f}, max: {r.kn_max:,.0f}\n")
        f.write(f"Mate positions: {r.mate_count:,} ({100 * r.mate_count / total_rows:.2f}%)\n")
        f.write(f"PV length — mean: {pv_mean:.2f}\n")
        f.write(f"White to move: {r.white_to_move:,}, Black to move: {r.black_to_move:,}\n")
    print(f"Text report saved to {report_path}")
