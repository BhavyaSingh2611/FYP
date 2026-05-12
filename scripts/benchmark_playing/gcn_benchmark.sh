#!/usr/bin/env bash
set -euo pipefail

BACKBONE="gcn"
STOCKFISH="/scratch/bs01346/FYP/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

echo "[${BACKBONE}] Benchmark 1/2: 10M samples, 50 epochs"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_10M_e50/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_10M_e50_best

echo "[${BACKBONE}] Benchmark 2/2: 100M samples, 15 epochs"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_100M_e15/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_100M_e15_best

echo "[${BACKBONE}] All playing benchmarks completed."