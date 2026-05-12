#!/usr/bin/env bash
set -euo pipefail

BACKBONE="piece_transformer"
DATA_PATH="data/puzzles/*.parquet"

echo "[${BACKBONE}] Puzzle Benchmark 1/5: 10M samples, 50 epochs"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_10M_e50/training/${BACKBONE}/best.pt --data-path ${DATA_PATH} --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_10M_e50_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_10M_e50_best/best_puzzle.png

echo "[${BACKBONE}] Puzzle Benchmark 2/5: 100M samples, 15 epochs"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_100M_e15/training/${BACKBONE}/best.pt --data-path ${DATA_PATH} --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_100M_e15_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_100M_e15_best/best_puzzle.png

echo "[${BACKBONE}] Puzzle Benchmark 3/5: 200M samples, 10 epochs"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_200M_e10/training/${BACKBONE}/best.pt --data-path ${DATA_PATH} --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_200M_e10_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_200M_e10_best/best_puzzle.png

echo "[${BACKBONE}] Puzzle Benchmark 4/5: 500M samples, 5 epochs"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_500M_e5/training/${BACKBONE}/best.pt --data-path ${DATA_PATH} --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_500M_e5_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_500M_e5_best/best_puzzle.png

echo "[${BACKBONE}] Puzzle Benchmark 5/5: 1B samples, 1 epoch"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_1000M_e1/training/${BACKBONE}/best.pt --data-path ${DATA_PATH} --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_1000M_e1_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_1000M_e1_best/best_puzzle.png

echo "[${BACKBONE}] All puzzle benchmarks completed."