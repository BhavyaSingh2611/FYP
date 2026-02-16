#!/usr/bin/env bash
# Launch supervised training across otter1..6, one model per machine.
# Each machine has a tmux session called "training" already running.

set -euo pipefail

REMOTE_DIR="/scratch/bs01346/FYP"

declare -A MODELS=(
  [otter1]="convnet"
  [otter2]="resnet"
  [otter3]="square_transformer"
  [otter4]="piece_transformer"
  [otter5]="gcn"
  [otter6]="gat"
)

for host in otter{1..6}; do
  model="${MODELS[$host]}"
  run_name="${model}_50_10M"
  cmd="cd ${REMOTE_DIR} && source .venv/bin/activate && python3 scripts/train.py --config config/config.yaml --name ${run_name} supervised --model ${model} --epochs 50 --num-samples 10000000"

  echo "[$host] Launching ${model}..."
  ssh -o StrictHostKeyChecking=no "$host" \
    "tmux send-keys -t training '${cmd}' Enter"
done

echo ""
echo "All 6 training runs launched:"
for host in otter{1..6}; do
  echo "  $host -> ${MODELS[$host]}"
done
echo ""
echo "Monitor with:  ssh otterN 'tmux attach -t training'"
