#!/usr/bin/env bash
set -euo pipefail

echo "[piece_transformer] Iteration 1/5: 10M samples, 50 epochs"
python3 scripts/train.py --model piece_transformer --epochs 50 --num-samples 10000000 --name piece_transformer_10M_e50

echo "[piece_transformer] Iteration 2/5: 100M samples, 15 epochs"
python3 scripts/train.py --model piece_transformer --epochs 15 --num-samples 100000000 --name piece_transformer_100M_e15

echo "[piece_transformer] Iteration 3/5: 200M samples, 10 epochs"
python3 scripts/train.py --model piece_transformer --epochs 10 --num-samples 200000000 --name piece_transformer_200M_e10

echo "[piece_transformer] Iteration 4/5: 500M samples, 5 epochs"
python3 scripts/train.py --model piece_transformer --epochs 5 --num-samples 500000000 --name piece_transformer_500M_e5

echo "[piece_transformer] Iteration 5/5: 1B samples, 1 epoch"
python3 scripts/train.py --model piece_transformer --epochs 1 --num-samples 1000000000 --name piece_transformer_1000M_e1

echo "[piece_transformer] All iterations completed."