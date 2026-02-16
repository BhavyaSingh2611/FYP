#!/usr/bin/env bash
# Launch Stockfish RL training across otter1..6, one model per machine.
# Each machine has a tmux session called "training" already running.
# Uses supervised checkpoints from runs/50_10M/*.pt

set -euo pipefail

REMOTE_DIR="/scratch/bs01346/FYP"
RUN_NAME="50_10M"
GAMES=40
ITERATIONS=10
STOCKFISH="${REMOTE_DIR}/stockfish/stockfish/stockfish-ubuntu-x86-64-avxvnni"

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
  cmd="cd ${REMOTE_DIR} && source .venv/bin/activate && python3 scripts/train.py --name 50_10M_sf --device cuda stockfish-rl --model ${model} --checkpoint runs/${RUN_NAME}/${model}.pt --games ${GAMES} --iterations ${ITERATIONS} --curriculum --stockfish-path ${STOCKFISH}"

  echo "[$host] Launching stockfish-rl: ${model}..."
  ssh -o StrictHostKeyChecking=no "$host" \
    "tmux send-keys -t training '${cmd}' Enter"
done

echo ""
echo "All 6 Stockfish RL runs launched:"
for host in otter{1..6}; do
  echo "  $host -> ${MODELS[$host]}"
done
echo ""
echo "Monitor with:  ssh otterN 'tmux attach -t training'"
