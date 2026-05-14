#!/usr/bin/env bash
set -euo pipefail

echo "[square_transformer] Puzzle Benchmark 1/5: 10M samples, 50 epochs"
python3 scripts/puzzle_benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_10M_e50.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/square_transformer/square_transformer_10M_e50
mv runs/elo_evaluation.png runs/puzzle_evaluation/square_transformer/square_transformer_10M_e50/best_puzzle.png

echo "[square_transformer] Puzzle Benchmark 2/5: 100M samples, 15 epochs"
python3 scripts/puzzle_benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_100M_e15.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/square_transformer/square_transformer_100M_e15
mv runs/elo_evaluation.png runs/puzzle_evaluation/square_transformer/square_transformer_100M_e15/best_puzzle.png

echo "[square_transformer] Puzzle Benchmark 3/5: 200M samples, 10 epochs"
python3 scripts/puzzle_benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_200M_e10.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/square_transformer/square_transformer_200M_e10
mv runs/elo_evaluation.png runs/puzzle_evaluation/square_transformer/square_transformer_200M_e10/best_puzzle.png

echo "[square_transformer] Puzzle Benchmark 4/5: 500M samples, 5 epochs"
python3 scripts/puzzle_benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_500M_e5.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/square_transformer/square_transformer_500M_e5
mv runs/elo_evaluation.png runs/puzzle_evaluation/square_transformer/square_transformer_500M_e5/best_puzzle.png

echo "[square_transformer] Puzzle Benchmark 5/5: 1B samples, 1 epoch"
python3 scripts/puzzle_benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_1000M_e1.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/square_transformer/square_transformer_1000M_e1
mv runs/elo_evaluation.png runs/puzzle_evaluation/square_transformer/square_transformer_1000M_e1/best_puzzle.png
