#!/bin/bash

# Configuration
USER="bs01346"
# Assuming the nodes list from cluster.sh
ALL_NODES=("otter1" "otter2" "otter3" "otter4" "otter5" "otter6")
NODE_NAMES=("otter1" "otter2" "otter3" "otter4" "otter5" "otter6")
TEMP_DIR="/scratch/temp"
DATA_DIR="/scratch/bs01346/FYP/data"

# 1. Safety Check: Is the SSH Agent running?
if [ -z "$SSH_AUTH_SOCK" ]; then
  echo "❌ Error: SSH Agent is not running. Run: eval \$(ssh-agent -s) && ssh-add"
  exit 1
fi

for i in "${!ALL_NODES[@]}"; do
  NODE_HOST=${ALL_NODES[$i]}
  NODE_NAME=${NODE_NAMES[$i]}

  echo "🚀 Starting parallel process for $NODE_NAME ($NODE_HOST)..."

  (
    ssh -o StrictHostKeyChecking=no "$NODE_HOST" "
      mkdir -p $TEMP_DIR && \
      mkdir -p $DATA_DIR && \
      curl -sL -o $TEMP_DIR/lichess-evals-stripped.zip https://www.kaggle.com/api/v1/datasets/download/bhavyasingh2611/lichess-evals-stripped && \
      unzip -qo $TEMP_DIR/lichess-evals-stripped.zip -d $TEMP_DIR/temp_evals && \
      mv $TEMP_DIR/temp_evals/*.parquet $DATA_DIR/chess_eval.parquet && \
      rm $TEMP_DIR/lichess-evals-stripped.zip && \
      rm -rf $TEMP_DIR/temp_evals
    "

    if [ $? -eq 0 ]; then
      echo "✅ $NODE_NAME: Success"
    else
      echo "❌ $NODE_NAME: Failed"
    fi
  ) &
done

echo "⏳ Waiting for all background downloads to complete..."
wait

echo "Download script complete."
