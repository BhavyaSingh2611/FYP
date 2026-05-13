#!/usr/bin/env bash
set -euo pipefail

STOCKFISH="/scratch/bs01346/FYP/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

echo "[gat] Benchmark 1/2: 10M samples, 50 epochs"
python3 scripts/benchmark.py --backbone gat --weights runs/models/gat/gat_10M_e50.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/gat/gat_10M_e50

echo "[gat] Benchmark 2/2: 100M samples, 15 epochs"
python3 scripts/benchmark.py --backbone gat --weights runs/models/gat/gat_100M_e15.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/gat/gat_100M_e15
