#!/usr/bin/env bash
set -euo pipefail

STOCKFISH="/scratch/bs01346/FYP/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

echo "[gcn] Benchmark 1/2: 10M samples, 50 epochs"
python3 scripts/benchmark.py --backbone gcn --weights runs/models/gcn/gcn_10M_e50.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/gcn/gcn_10M_e50

echo "[gcn] Benchmark 2/2: 100M samples, 15 epochs"
python3 scripts/benchmark.py --backbone gcn --weights runs/models/gcn/gcn_100M_e15.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/gcn/gcn_100M_e15
