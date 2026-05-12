#!/usr/bin/env python3
import argparse
import logging
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path


def discover_models(models_dir: Path | None, models_glob: str | None, backbone: str | None) -> list[Path]:
    if models_glob:
        return sorted(Path(path) for path in glob(models_glob))

    if backbone:
        pattern = f"runs/{backbone}_*/training/{backbone}/best.pt"
        return sorted(Path(path) for path in glob(pattern))

    if models_dir:
        return sorted(models_dir.glob("*/*.pt"))

    return []


def run_command(cmd: list[str], dry_run: bool) -> None:
    logging.info("Running: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_playing_benchmark(
    python_exe: str,
    benchmark_script: Path,
    backbone: str,
    weights_path: Path,
    output_dir: Path,
    stockfish_path: str,
    games: int,
    levels: str | None,
    workers: int,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(benchmark_script),
        "--backbone",
        backbone,
        "--weights",
        str(weights_path),
        "--stockfish",
        stockfish_path,
        "--games",
        str(games),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(workers),
    ]
    if levels:
        cmd.extend(["--levels", levels])
    run_command(cmd, dry_run)


def run_puzzle_benchmark(
    python_exe: str,
    puzzle_script: Path,
    backbone: str,
    weights_path: Path,
    data_path: str,
    min_elo: int,
    max_elo: int,
    step_elo: int,
    puzzles_per_bracket: int,
    output_dir: Path,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(puzzle_script),
        "--backbone",
        backbone,
        "--weights",
        str(weights_path),
        "--data-path",
        data_path,
        "--min-elo",
        str(min_elo),
        "--max-elo",
        str(max_elo),
        "--step-elo",
        str(step_elo),
        "--puzzles-per-bracket",
        str(puzzles_per_bracket),
    ]
    run_command(cmd, dry_run)

    if dry_run:
        return

    generated_plot = Path("runs/elo_evaluation.png")
    if not generated_plot.exists():
        logging.warning("Puzzle plot not found at %s", generated_plot)
        return

    target_plot = output_dir / f"{weights_path.stem}_puzzle.png"
    shutil.move(str(generated_plot), str(target_plot))
    logging.info("Saved puzzle plot to %s", target_plot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run puzzle + playing benchmarks for all models in runs/models.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="runs/models",
        help="Directory containing model subfolders with .pt files.",
    )
    parser.add_argument(
        "--models-glob",
        type=str,
        default=None,
        help="Glob pattern for model weights (.pt). Overrides --models-dir.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone name to auto-discover runs/<backbone>_*/training/<backbone>/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluation_all",
        help="Output directory for playing benchmark artifacts.",
    )
    parser.add_argument(
        "--puzzle-output-dir",
        type=str,
        default="runs/puzzle_evaluation",
        help="Output directory for puzzle benchmark plots.",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=4,
        help="Games per difficulty level (use even number).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated level indices (0-9) for playing benchmark.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel threads (each spawns its own Stockfish).",
    )
    parser.add_argument(
        "--puzzle-data",
        type=str,
        default="data/puzzles/*.parquet",
        help="Glob pattern for Lichess puzzle parquet files.",
    )
    parser.add_argument("--min-elo", type=int, default=800, help="Starting Elo bracket.")
    parser.add_argument("--max-elo", type=int, default=2800, help="Ending Elo bracket.")
    parser.add_argument("--step-elo", type=int, default=100, help="Step size between Elo brackets.")
    parser.add_argument(
        "--puzzles-per-bracket",
        type=int,
        default=20,
        help="Number of puzzles to evaluate per bracket.",
    )
    parser.add_argument(
        "--skip-playing",
        action="store_true",
        help="Skip playing benchmark.",
    )
    parser.add_argument(
        "--skip-puzzles",
        action="store_true",
        help="Skip puzzle benchmark.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run benchmark scripts.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    models_dir = Path(args.models_dir) if args.models_dir else None
    if models_dir and not models_dir.exists() and not args.models_glob and not args.backbone:
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models = discover_models(models_dir, args.models_glob, args.backbone)
    if not models:
        raise FileNotFoundError("No .pt files found for the selected discovery options.")

    benchmark_script = Path("scripts/benchmark.py")
    puzzle_script = Path("scripts/benchmark_elo.py")

    for weights_path in models:
        run_name = (
            weights_path.parent.parent.name
            if weights_path.parent.name == weights_path.parent.parent.name.split("_")[0]
            or weights_path.parent.parent.name.startswith(weights_path.parent.name)
            else weights_path.parent.name
        )
        backbone = args.backbone if args.backbone else weights_path.parent.name
        logging.info("=== Benchmarking %s (%s) ===", weights_path.name, backbone)

        if not args.skip_playing:
            output_dir = Path(args.output_dir) / backbone / f"{run_name}_{weights_path.stem}"
            run_playing_benchmark(
                python_exe=args.python,
                benchmark_script=benchmark_script,
                backbone=backbone,
                weights_path=weights_path,
                output_dir=output_dir,
                stockfish_path=args.stockfish,
                games=args.games,
                levels=args.levels,
                workers=args.workers,
                dry_run=args.dry_run,
            )

        if not args.skip_puzzles:
            puzzle_output_dir = Path(args.puzzle_output_dir) / backbone / f"{run_name}_{weights_path.stem}"
            run_puzzle_benchmark(
                python_exe=args.python,
                puzzle_script=puzzle_script,
                backbone=backbone,
                weights_path=weights_path,
                data_path=args.puzzle_data,
                min_elo=args.min_elo,
                max_elo=args.max_elo,
                step_elo=args.step_elo,
                puzzles_per_bracket=args.puzzles_per_bracket,
                output_dir=puzzle_output_dir,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
