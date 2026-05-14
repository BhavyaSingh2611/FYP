#!/usr/bin/env bash
set -euo pipefail

echo "[gcn] Puzzle Benchmark 1/2: 10M samples, 50 epochs"
python3 scripts/puzzle_benchmark.py --backbone gcn --weights runs/models/gcn/gcn_10M_e50.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/gcn/gcn_10M_e50
mv runs/elo_evaluation.png runs/puzzle_evaluation/gcn/gcn_10M_e50/best_puzzle.png

echo "[gcn] Puzzle Benchmark 2/2: 100M samples, 15 epochs"
python3 scripts/puzzle_benchmark.py --backbone gcn --weights runs/models/gcn/gcn_100M_e15.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/gcn/gcn_100M_e15
mv runs/elo_evaluation.png runs/puzzle_evaluation/gcn/gcn_100M_e15/best_puzzle.png
