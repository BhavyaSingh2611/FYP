#!/usr/bin/env python3
"""
Dataset preparation pipeline.

Usage:
  python scripts/prepare_dataset.py clean-dedup --input data/ --output data/deduped/
  python scripts/prepare_dataset.py process --input data/deduped/ --output data/staged/ --openings data/openings.csv --tablebase stockfish/syzygy/ --workers 4
  python scripts/prepare_dataset.py shuffle --input data/staged/ --output data/processed/ --shard-size 50000000
"""
import argparse
import csv
import multiprocessing
from pathlib import Path

import chess
import chess.syzygy
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

OUTPUT_SCHEMA = pa.schema([
    pa.field("fen", pa.string()),
    pa.field("best_move", pa.string()),
    pa.field("cp", pa.int16()),
    pa.field("mate", pa.int8()),
    pa.field("depth", pa.uint8()),
    pa.field("value", pa.float32()),
])


# ---------------------------------------------------------------------------
# Step 0 — Opening book set
# ---------------------------------------------------------------------------

def build_opening_set(openings_csv: Path) -> set[str]:
    epds: set[str] = set()
    with open(openings_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uci_moves = row["uci"].strip()
            if not uci_moves:
                continue
            board = chess.Board()
            epds.add(" ".join(board.fen().split()[:4]))
            for uci in uci_moves.split():
                try:
                    board.push_uci(uci)
                except ValueError:
                    break
                epds.add(" ".join(board.fen().split()[:4]))
    print(f"Opening book: {len(epds):,} unique positions")
    return epds


# ---------------------------------------------------------------------------
# Step 1 — Clean & global dedup (DuckDB)
# ---------------------------------------------------------------------------

def cmd_clean_dedup(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_buckets = getattr(args, "buckets", 32)
    pattern = str(input_dir / "train-*.parquet")
    temp_path = str(output_dir / "_filtered_temp.parquet")

    con = duckdb.connect()
    con.execute("SET temp_directory = '/tmp/duckdb_tmp'")
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 2")

    # Pass 1: Filter and pre-compute norm_fen (skip if temp file exists)
    if Path(temp_path).exists() and Path(temp_path).stat().st_size > 0:
        print(f"Pass 1: Skipped — {temp_path} already exists")
        filtered_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{temp_path}')"
        ).fetchone()[0]
        print(f"Filtered rows: {filtered_count:,}")
    else:
        print(f"Source pattern: {pattern}")
        con.execute(f"""
            CREATE VIEW raw AS
            SELECT * FROM read_parquet('{pattern}')
        """)
        total = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
        print(f"Total raw rows: {total:,}")

        print("Pass 1: Filtering ...")
        con.execute(f"""
            COPY (
                SELECT
                    fen, line, cp, mate, depth, knodes,
                    regexp_replace(fen, '(\\s\\S+){{2}}$', '') AS norm_fen
                FROM raw
                WHERE depth > 18
                  AND knodes IS NOT NULL AND knodes > 0
                  AND fen IS NOT NULL AND fen != ''
                  AND line IS NOT NULL AND line != ''
                  AND (cp IS NOT NULL OR mate IS NOT NULL)
                  AND (cp IS NULL OR ABS(cp) <= 4000)
            ) TO '{temp_path}'
            (FORMAT PARQUET, ROW_GROUP_SIZE 100000)
        """)
        filtered_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{temp_path}')"
        ).fetchone()[0]
        print(f"After filtering: {filtered_count:,}")

    # Pass 2: Dedup in buckets (partition by hash of norm_fen so each
    # bucket's hash table fits in memory)
    parts_dir = output_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)

    existing_parts = sorted(parts_dir.glob("part-*.parquet"))
    start_bucket = len(existing_parts)
    if start_bucket > 0:
        print(f"Pass 2: Resuming from bucket {start_bucket}/{num_buckets}")

    for bucket in range(start_bucket, num_buckets):
        part_path = parts_dir / f"part-{bucket:04d}.parquet"
        print(f"Pass 2: Dedup bucket {bucket + 1}/{num_buckets} ...")
        con.execute(f"""
            COPY (
                SELECT
                    first(fen ORDER BY depth DESC, knodes DESC)    AS fen,
                    first(line ORDER BY depth DESC, knodes DESC)   AS line,
                    first(cp ORDER BY depth DESC, knodes DESC)     AS cp,
                    first(mate ORDER BY depth DESC, knodes DESC)   AS mate,
                    first(depth ORDER BY depth DESC, knodes DESC)  AS depth,
                    first(knodes ORDER BY depth DESC, knodes DESC) AS knodes
                FROM read_parquet('{temp_path}')
                WHERE hash(norm_fen) % {num_buckets} = {bucket}
                GROUP BY norm_fen
            ) TO '{part_path}'
            (FORMAT PARQUET, ROW_GROUP_SIZE 100000)
        """)

    # Merge all parts into final output
    parts_pattern = str(parts_dir / "part-*.parquet")
    output_path = str(output_dir / "deduped.parquet")
    print("Merging parts ...")
    con.execute(f"""
        COPY (
            SELECT fen, line, cp, mate, depth, knodes
            FROM read_parquet('{parts_pattern}')
        ) TO '{output_path}'
        (FORMAT PARQUET, ROW_GROUP_SIZE 100000)
    """)

    deduped_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    print(f"After dedup: {deduped_count:,} ({filtered_count - deduped_count:,} duplicates removed)")

    # Cleanup temp files
    for p in parts_dir.glob("part-*.parquet"):
        p.unlink()
    parts_dir.rmdir()
    Path(temp_path).unlink(missing_ok=True)

    con.close()
    print("Clean-dedup complete.")


# ---------------------------------------------------------------------------
# Step 2 — Process (PV unroll, phase classify, tablebase, per-file)
# ---------------------------------------------------------------------------

def classify_phase(board: chess.Board, norm_fen: str, opening_set: set[str]) -> str:
    if norm_fen in opening_set:
        return "opening"

    total_material = sum(
        PIECE_VALUES.get(piece.piece_type, 0)
        for piece in board.piece_map().values()
        if piece.piece_type != chess.KING
    )
    has_queens = bool(
        board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK)
    )

    if total_material < 16 or (not has_queens and total_material < 10):
        return "endgame"
    return "midgame"


def compute_value(cp: int | None, mate: int | None, board: chess.Board) -> float:
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    if mate is not None:
        return sign * (1.0 if mate > 0 else -1.0)
    if cp is not None:
        return sign * max(-1.0, min(1.0, cp / 1000.0))
    return 0.0


def find_best_tb_move(board: chess.Board, tablebase: chess.syzygy.Tablebase) -> chess.Move | None:
    root_wdl = tablebase.probe_wdl(board)
    best_move = None
    best_dtz = None

    for move in board.legal_moves:
        board.push(move)
        try:
            child_dtz = tablebase.probe_dtz(board)
        except chess.syzygy.MissingTableError:
            board.pop()
            continue
        board.pop()

        if root_wdl > 0:
            if child_dtz is not None and child_dtz < 0:
                if best_dtz is None or child_dtz > best_dtz:
                    best_dtz = child_dtz
                    best_move = move
        elif root_wdl < 0:
            if child_dtz is not None and child_dtz > 0:
                if best_dtz is None or child_dtz < best_dtz:
                    best_dtz = child_dtz
                    best_move = move
        else:
            if child_dtz is not None and child_dtz == 0:
                best_move = move
                break

    return best_move


def process_single_file(
    file_path: Path,
    output_dir: Path,
    opening_set: set[str],
    tablebase_path: str | None,
):
    tb = None
    if tablebase_path and Path(tablebase_path).is_dir():
        try:
            tb = chess.syzygy.open_tablebase(tablebase_path)
        except Exception:
            tb = None

    seen_fens: set[str] = set()
    buffers: dict[str, list[dict]] = {"opening": [], "midgame": [], "endgame": []}

    pf = pq.ParquetFile(file_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        fens = table.column("fen").to_pylist()
        lines = table.column("line").to_pylist()
        cps = table.column("cp").to_pylist()
        mates = table.column("mate").to_pylist()
        depths = table.column("depth").to_pylist()

        for i in range(len(fens)):
            fen_str = fens[i]
            line_str = lines[i]
            cp = cps[i]
            mate = mates[i]
            depth = depths[i]

            if not fen_str or not line_str:
                continue

            try:
                board = chess.Board(fen_str)
            except ValueError:
                continue

            pv_moves = line_str.split()
            if not pv_moves:
                continue

            def emit(board: chess.Board, fen: str, best_move: str, cp, mate, depth):
                norm = " ".join(board.fen().split()[:4])
                if norm in seen_fens:
                    return
                seen_fens.add(norm)

                phase = classify_phase(board, norm, opening_set)
                value = compute_value(cp, mate, board)

                if tb and len(board.piece_map()) <= 5:
                    try:
                        wdl = tb.probe_wdl(board)
                        tb_move = find_best_tb_move(board, tb)
                        if tb_move:
                            best_move = tb_move.uci()
                        if wdl > 0:
                            cp, mate = None, None
                            value = 1.0
                        elif wdl < 0:
                            cp, mate = None, None
                            value = -1.0
                        else:
                            cp, mate = 0, None
                            value = 0.0
                    except chess.syzygy.MissingTableError:
                        pass

                buffers[phase].append({
                    "fen": fen,
                    "best_move": best_move,
                    "cp": cp,
                    "mate": mate,
                    "depth": depth,
                    "value": value,
                })

            emit(board, fen_str, pv_moves[0], cp, mate, depth)

            if depth is not None and depth >= 15 and len(pv_moves) >= 4:
                unroll_cp = cp
                unroll_mate = mate
                unroll_board = board.copy()

                for j in range(1, len(pv_moves)):
                    try:
                        unroll_board.push_uci(pv_moves[j - 1])
                    except (ValueError, chess.IllegalMoveError):
                        break

                    unroll_cp = -unroll_cp if unroll_cp is not None else None
                    unroll_mate = -unroll_mate if unroll_mate is not None else None

                    emit(
                        unroll_board,
                        unroll_board.fen(),
                        pv_moves[j],
                        unroll_cp,
                        unroll_mate,
                        depth,
                    )

    file_stem = file_path.stem
    for phase, rows in buffers.items():
        if not rows:
            continue
        phase_dir = output_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pydict(
            {
                "fen": [r["fen"] for r in rows],
                "best_move": [r["best_move"] for r in rows],
                "cp": [r["cp"] for r in rows],
                "mate": [r["mate"] for r in rows],
                "depth": [r["depth"] for r in rows],
                "value": [r["value"] for r in rows],
            },
            schema=OUTPUT_SCHEMA,
        )
        pq.write_table(table, phase_dir / f"{file_stem}.parquet", row_group_size=100_000)

    total = sum(len(v) for v in buffers.values())
    counts = {k: len(v) for k, v in buffers.items() if v}
    print(f"  {file_path.name}: {total:,} positions {counts}")

    if tb:
        tb.close()


def _process_worker(args_tuple):
    process_single_file(*args_tuple)


def cmd_process(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    openings_csv = Path(args.openings)
    tablebase_path = args.tablebase

    opening_set = build_opening_set(openings_csv)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Processing {len(files)} file(s) with {args.workers} worker(s)...")

    work_items = [
        (f, output_dir, opening_set, tablebase_path)
        for f in files
    ]

    if args.workers <= 1:
        for item in work_items:
            _process_worker(item)
    else:
        with multiprocessing.Pool(args.workers) as pool:
            list(pool.imap_unordered(_process_worker, work_items))

    print("Process complete.")


# ---------------------------------------------------------------------------
# Step 3 — Global shuffle & shard
# ---------------------------------------------------------------------------

def cmd_shuffle(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    shard_size = args.shard_size

    for phase in ["opening", "midgame", "endgame"]:
        phase_in = input_dir / phase
        if not phase_in.is_dir():
            print(f"Skipping {phase} (no directory)")
            continue

        files = sorted(phase_in.glob("*.parquet"))
        if not files:
            print(f"Skipping {phase} (no files)")
            continue

        print(f"\n{'='*60}")
        print(f"Shuffling phase: {phase}")
        print(f"{'='*60}")

        tables = []
        total_rows = 0
        for f in tqdm(files, desc=f"Reading {phase}"):
            t = pq.read_table(f, schema=OUTPUT_SCHEMA)
            tables.append(t)
            total_rows += len(t)

        print(f"Total rows: {total_rows:,}")

        combined = pa.concat_tables(tables)
        del tables

        permutation = np.random.permutation(total_rows)

        phase_out = output_dir / phase
        phase_out.mkdir(parents=True, exist_ok=True)

        num_shards = (total_rows + shard_size - 1) // shard_size
        print(f"Writing {num_shards} shard(s) of ~{shard_size:,} rows each...")

        for shard_idx in tqdm(range(num_shards), desc=f"Writing {phase}"):
            start = shard_idx * shard_size
            end = min(start + shard_size, total_rows)
            indices = permutation[start:end]

            shard = combined.take(indices)
            out_path = phase_out / f"shard-{shard_idx:04d}.parquet"
            pq.write_table(shard, out_path, row_group_size=100_000)

        del combined
        print(f"{phase}: {num_shards} shard(s) written to {phase_out}")

    print("\nShuffle complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dataset preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("clean-dedup", help="Filter bad rows and global dedup via DuckDB")
    p1.add_argument("--input", required=True, help="Directory containing train-*.parquet files")
    p1.add_argument("--output", required=True, help="Output directory for deduped parquet")
    p1.add_argument("--buckets", type=int, default=32, help="Number of hash buckets for dedup (more = less memory)")
    p1.set_defaults(func=cmd_clean_dedup)

    p2 = subparsers.add_parser("process", help="PV unroll, phase classify, tablebase correction")
    p2.add_argument("--input", required=True, help="Directory containing deduped parquet files")
    p2.add_argument("--output", required=True, help="Output directory for staged phase files")
    p2.add_argument("--openings", default="data/openings.csv", help="Path to openings CSV")
    p2.add_argument("--tablebase", default=None, help="Path to Syzygy tablebase directory")
    p2.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    p2.set_defaults(func=cmd_process)

    p3 = subparsers.add_parser("shuffle", help="Global shuffle and shard per phase")
    p3.add_argument("--input", required=True, help="Directory containing staged phase directories")
    p3.add_argument("--output", required=True, help="Output directory for final shards")
    p3.add_argument("--shard-size", type=int, default=50_000_000, help="Rows per shard")
    p3.set_defaults(func=cmd_shuffle)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
