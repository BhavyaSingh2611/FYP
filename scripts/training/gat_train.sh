#!/usr/bin/env bash
set -euo pipefail

echo "[gat] Iteration 1/5: 10M samples, 50 epochs"
python3 scripts/train.py --model gat --epochs 50 --num-samples 10000000 --batch-size 1024 --name gat_10M_e50

echo "[gat] Iteration 2/5: 100M samples, 15 epochs"
python3 scripts/train.py --model gat --epochs 15 --num-samples 100000000 --batch-size 1024 --name gat_100M_e15

echo "[gat] Iteration 3/5: 200M samples, 10 epochs"
python3 scripts/train.py --model gat --epochs 10 --num-samples 200000000 --batch-size 1024 --name gat_200M_e10

echo "[gat] Iteration 4/5: 500M samples, 5 epochs"
python3 scripts/train.py --model gat --epochs 5 --num-samples 500000000 --batch-size 1024 --name gat_500M_e5

echo "[gat] Iteration 5/5: 1B samples, 1 epoch"
python3 scripts/train.py --model gat --epochs 1 --num-samples 1000000000 --batch-size 1024 --name gat_1000M_e1

echo "[gat] All iterations completed."