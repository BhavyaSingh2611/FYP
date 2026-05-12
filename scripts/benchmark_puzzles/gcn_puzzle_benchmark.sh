#!/usr/bin/env bash
set -euo pipefail

BACKBONE="gcn"

echo "[${BACKBONE}] Puzzle Benchmark 1/2: 10M samples, 50 epochs"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_10M_e50/training/${BACKBONE}/best.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_10M_e50_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_10M_e50_best/best_puzzle.png

echo "[${BACKBONE}] Puzzle Benchmark 2/2: 100M samples, 15 epochs"
python3 scripts/benchmark_elo.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_100M_e15/training/${BACKBONE}/best.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_100M_e15_best
mv runs/elo_evaluation.png runs/puzzle_evaluation/${BACKBONE}/${BACKBONE}_100M_e15_best/best_puzzle.png

echo "[${BACKBONE}] All puzzle benchmarks completed."