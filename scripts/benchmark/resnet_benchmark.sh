#!/usr/bin/env bash
set -euo pipefail

STOCKFISH="/scratch/bs01346/FYP/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

echo "[resnet] Benchmark 1/5: 10M samples, 50 epochs"
python3 scripts/benchmark.py --backbone resnet --weights runs/models/resnet/resnet_10M_e50.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/resnet/resnet_10M_e50

echo "[resnet] Benchmark 2/5: 100M samples, 15 epochs"
python3 scripts/benchmark.py --backbone resnet --weights runs/models/resnet/resnet_100M_e15.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/resnet/resnet_100M_e15

echo "[resnet] Benchmark 3/5: 200M samples, 10 epochs"
python3 scripts/benchmark.py --backbone resnet --weights runs/models/resnet/resnet_200M_e10.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/resnet/resnet_200M_e10

echo "[resnet] Benchmark 4/5: 500M samples, 5 epochs"
python3 scripts/benchmark.py --backbone resnet --weights runs/models/resnet/resnet_500M_e5.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/resnet/resnet_500M_e5

echo "[resnet] Benchmark 5/5: 1B samples, 1 epoch"
python3 scripts/benchmark.py --backbone resnet --weights runs/models/resnet/resnet_1000M_e1.pt --stockfish ${STOCKFISH} --games 8 --workers 4 --output-dir runs/evaluation_all/resnet/resnet_1000M_e1
