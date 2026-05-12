#!/usr/bin/env bash
set -euo pipefail

BACKBONE="square_transformer"
STOCKFISH="/scratch/bs01346/FYP/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

echo "[${BACKBONE}] Benchmark 1/5: 10M samples, 50 epochs"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_10M_e50/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_10M_e50_best

echo "[${BACKBONE}] Benchmark 2/5: 100M samples, 15 epochs"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_100M_e15/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_100M_e15_best

echo "[${BACKBONE}] Benchmark 3/5: 200M samples, 10 epochs"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_200M_e10/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_200M_e10_best

echo "[${BACKBONE}] Benchmark 4/5: 500M samples, 5 epochs"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_500M_e5/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_500M_e5_best

echo "[${BACKBONE}] Benchmark 5/5: 1B samples, 1 epoch"
python3 scripts/benchmark.py --backbone ${BACKBONE} --weights runs/${BACKBONE}_1000M_e1/training/${BACKBONE}/best.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/${BACKBONE}/${BACKBONE}_1000M_e1_best

echo "[${BACKBONE}] All playing benchmarks completed."