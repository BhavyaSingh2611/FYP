#!/usr/bin/env bash
set -euo pipefail

STOCKFISH="/scratch/bs01346/FYP/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

echo "[square_transformer] Benchmark 1/5: 10M samples, 50 epochs"
python3 scripts/benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_10M_e50.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/square_transformer/square_transformer_10M_e50

echo "[square_transformer] Benchmark 2/5: 100M samples, 15 epochs"
python3 scripts/benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_100M_e15.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/square_transformer/square_transformer_100M_e15

echo "[square_transformer] Benchmark 3/5: 200M samples, 10 epochs"
python3 scripts/benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_200M_e10.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/square_transformer/square_transformer_200M_e10

echo "[square_transformer] Benchmark 4/5: 500M samples, 5 epochs"
python3 scripts/benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_500M_e5.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/square_transformer/square_transformer_500M_e5

echo "[square_transformer] Benchmark 5/5: 1B samples, 1 epoch"
python3 scripts/benchmark.py --backbone square_transformer --weights runs/models/square_transformer/square_transformer_1000M_e1.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/square_transformer/square_transformer_1000M_e1
