#!/usr/bin/env bash
set -euo pipefail

echo "[resnet] Puzzle Benchmark 1/5: 10M samples, 50 epochs"
python3 scripts/puzzle_benchmark.py --backbone resnet --weights runs/models/resnet/resnet_10M_e50.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/resnet/resnet_10M_e50
mv runs/elo_evaluation.png runs/puzzle_evaluation/resnet/resnet_10M_e50/best_puzzle.png

echo "[resnet] Puzzle Benchmark 2/5: 100M samples, 15 epochs"
python3 scripts/puzzle_benchmark.py --backbone resnet --weights runs/models/resnet/resnet_100M_e15.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/resnet/resnet_100M_e15
mv runs/elo_evaluation.png runs/puzzle_evaluation/resnet/resnet_100M_e15/best_puzzle.png

echo "[resnet] Puzzle Benchmark 3/5: 200M samples, 10 epochs"
python3 scripts/puzzle_benchmark.py --backbone resnet --weights runs/models/resnet/resnet_200M_e10.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/resnet/resnet_200M_e10
mv runs/elo_evaluation.png runs/puzzle_evaluation/resnet/resnet_200M_e10/best_puzzle.png

echo "[resnet] Puzzle Benchmark 4/5: 500M samples, 5 epochs"
python3 scripts/puzzle_benchmark.py --backbone resnet --weights runs/models/resnet/resnet_500M_e5.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/resnet/resnet_500M_e5
mv runs/elo_evaluation.png runs/puzzle_evaluation/resnet/resnet_500M_e5/best_puzzle.png

echo "[resnet] Puzzle Benchmark 5/5: 1B samples, 1 epoch"
python3 scripts/puzzle_benchmark.py --backbone resnet --weights runs/models/resnet/resnet_1000M_e1.pt --min-elo 800 --max-elo 2800 --step-elo 100 --puzzles-per-bracket 40
mkdir -p runs/puzzle_evaluation/resnet/resnet_1000M_e1
mv runs/elo_evaluation.png runs/puzzle_evaluation/resnet/resnet_1000M_e1/best_puzzle.png
