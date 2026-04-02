#!/usr/bin/env bash
set -euo pipefail

echo "[resnet] Iteration 1/5: 10M samples, 50 epochs"
python3 scripts/train.py --database data/chess_eval.parquet --model resnet --epochs 50 --num-samples 10000000 --name resnet_10M_e50

echo "[resnet] Iteration 2/5: 100M samples, 15 epochs"
python3 scripts/train.py --database data/chess_eval.parquet --model resnet --epochs 15 --num-samples 100000000 --name resnet_100M_e15

echo "[resnet] Iteration 3/5: 200M samples, 10 epochs"
python3 scripts/train.py --database data/chess_eval.parquet --model resnet --epochs 10 --num-samples 200000000 --name resnet_200M_e10

echo "[resnet] Iteration 4/5: 500M samples, 5 epochs"
python3 scripts/train.py --database data/chess_eval.parquet --model resnet --epochs 5 --num-samples 500000000 --name resnet_500M_e5

echo "[resnet] Iteration 5/5: 1B samples, 1 epoch"
python3 scripts/train.py --database data/chess_eval.parquet --model resnet --epochs 1 --num-samples 1000000000 --name resnet_1000M_e1

echo "[resnet] All iterations completed."