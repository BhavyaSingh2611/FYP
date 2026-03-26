"""
Dataset analysis script – reads a single Parquet file in batches and
produces summary statistics + plots.

Parquet schema: f (FEN), b (best move UCI), v (value -1 to +1).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "chess_eval.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "data_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100_000
VALUE_HIST_EDGES = np.linspace(-1.0, 1.0, 201)

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    pf = pq.ParquetFile(DATA_FILE)
    num_batches = math.ceil(pf.metadata.num_rows / BATCH_SIZE)
    LOGGER.info("Reading %s  (%s rows)", DATA_FILE.name, f"{pf.metadata.num_rows:,}")

    total_rows = 0
    val_sum = 0.0
    val_sq_sum = 0.0
    val_count = 0
    val_min = float("inf")
    val_max = float("-inf")
    val_hist = np.zeros(len(VALUE_HIST_EDGES) - 1, dtype=np.int64)
    white_to_move = 0
    black_to_move = 0

    for batch in tqdm(
        pf.iter_batches(batch_size=BATCH_SIZE), total=num_batches, unit="batch"
    ):
        n = batch.num_rows
        total_rows += n

        v_arr = batch.column("v").to_numpy(zero_copy_only=False).astype(np.float64)
        mask_v = ~np.isnan(v_arr)
        v_valid = v_arr[mask_v]
        if len(v_valid):
            val_sum += v_valid.sum()
            val_sq_sum += (v_valid**2).sum()
            val_count += len(v_valid)
            val_min = min(val_min, float(v_valid.min()))
            val_max = max(val_max, float(v_valid.max()))
            val_hist += np.histogram(v_valid, bins=VALUE_HIST_EDGES)[0]

        fen_arr = batch.column("f").to_pylist()
        for fen in fen_arr:
            if fen and " w " in fen:
                white_to_move += 1
            elif fen:
                black_to_move += 1

    LOGGER.info("Total rows processed: %s", f"{total_rows:,}")

    val_mean = val_sum / val_count if val_count else 0
    val_std = ((val_sq_sum / val_count) - val_mean**2) ** 0.5 if val_count else 0
    LOGGER.info(
        """\
=================================================================
VALUE DISTRIBUTION (v)
=================================================================
  Count (non-null) : %s
  Mean             : %s
  Std              : %s
  Min              : %s
  Max              : %s""",
        f"{val_count:,}",
        f"{val_mean:+.4f}",
        f"{val_std:.4f}",
        f"{val_min:+.4f}",
        f"{val_max:+.4f}",
    )

    LOGGER.info(
        """\
=================================================================
ACTIVE COLOUR
=================================================================
  White to move : %s  (%s%%)
  Black to move : %s  (%s%%)""",
        f"{white_to_move:,}",
        f"{100 * white_to_move / total_rows:.2f}",
        f"{black_to_move:,}",
        f"{100 * black_to_move / total_rows:.2f}",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chess Dataset Analysis", fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0]
    centres = (VALUE_HIST_EDGES[:-1] + VALUE_HIST_EDGES[1:]) / 2
    ax.bar(centres, val_hist, width=0.01, color="steelblue", edgecolor="none")
    ax.set_xlabel("Value (v)")
    ax.set_ylabel("Count")
    ax.set_title("Value Distribution (-1 to +1)")
    ax.yaxis.set_major_formatter(ticker.EngFormatter())

    ax = axes[1]
    ax.pie(
        [white_to_move, black_to_move],
        labels=["White to move", "Black to move"],
        autopct="%1.1f%%",
        colors=["#f0d9b5", "#b58863"],
        textprops={"fontsize": 12},
    )
    ax.set_title("Active Colour Split")

    plt.tight_layout()
    out_path = OUT_DIR / "dataset_analysis.png"
    plt.savefig(out_path, dpi=150)
    LOGGER.info("Figure saved to %s", out_path)

    report_path = OUT_DIR / "dataset_analysis.txt"
    with open(report_path, "w") as f:
        f.write(f"Total rows: {total_rows:,}\n\n")
        f.write(
            f"Value — mean: {val_mean:+.4f}, std: {val_std:.4f}, "
            f"min: {val_min:+.4f}, max: {val_max:+.4f}, non-null: {val_count:,}\n"
        )
        f.write(f"White to move: {white_to_move:,}, Black to move: {black_to_move:,}\n")
    LOGGER.info("Text report saved to %s", report_path)
