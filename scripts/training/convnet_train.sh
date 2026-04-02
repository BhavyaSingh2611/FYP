#!/usr/bin/env bash
set -euo pipefail

echo "[convnet] Iteration 1/5: 100M samples, 20 epochs"
python scripts/train.py --model convnet --epochs 20 --num-samples 100000000 --name convnet_100M_e20

echo "[convnet] Iteration 2/5: 250M samples, 8 epochs"
python scripts/train.py --model convnet --epochs 8 --num-samples 250000000 --name convnet_250M_e8

echo "[convnet] Iteration 3/5: 500M samples, 4 epochs"
python scripts/train.py --model convnet --epochs 4 --num-samples 500000000 --name convnet_500M_e4

echo "[convnet] Iteration 4/5: 1B samples, 2 epochs"
python scripts/train.py --model convnet --epochs 2 --num-samples 1000000000 --name convnet_1000M_e2

echo "[convnet] Iteration 5/5: 1.88B samples, 1 epoch"
python scripts/train.py --model convnet --epochs 1 --num-samples 1880000000 --name convnet_1880M_e1

echo "[convnet] All iterations completed."