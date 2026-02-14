#!/usr/bin/env bash
# Launch self-play training across otter1..6, one model per machine.
# Each machine has a tmux session called "training" already running.
# Uses supervised checkpoints from runs/50_10M/*.pt

set -euo pipefail

REMOTE_DIR="/scratch/bs01346/FYP"
RUN_NAME="50_10M"
GAMES=50
ITERATIONS=5

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
  cmd="cd ${REMOTE_DIR} && source .venv/bin/activate && python3 scripts/self_play_all.py --run ${RUN_NAME} --models ${model} --games ${GAMES} --iterations ${ITERATIONS}"

  echo "[$host] Launching self-play: ${model}..."
  ssh -o StrictHostKeyChecking=no "$host" \
    "tmux send-keys -t training '${cmd}' Enter"
done

echo ""
echo "All 6 self-play runs launched:"
for host in otter{1..6}; do
  echo "  $host -> ${MODELS[$host]}"
done
echo ""
echo "Monitor with:  ssh otterN 'tmux attach -t training'"
