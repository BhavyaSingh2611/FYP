#!/bin/bash

# Configuration
USER="bs01346"
SOURCE_NODE="otter1"
DEST_HOSTS=(
  "otter21.eps.surrey.ac.uk" # otter2
  "otter22.eps.surrey.ac.uk" # otter3
  "otter23.eps.surrey.ac.uk" # otter4
  "otter24.eps.surrey.ac.uk" # otter5
  "otter25.eps.surrey.ac.uk" # otter6
)
DEST_NAMES=("otter2" "otter3" "otter4" "otter5" "otter6")
REMOTE_PATH="/scratch/bs01346/FYP"
PARENT_DIR="/scratch/bs01346"
REPORT_FILE="cluster_specs.txt"

# 1. Safety Check: Is the SSH Agent running?
if [ -z "$SSH_AUTH_SOCK" ]; then
  echo "❌ Error: SSH Agent is not running. Run: eval \$(ssh-agent -s) && ssh-add"
  exit 1
fi

>"$REPORT_FILE"

echo "--- PHASE 1: Internal Node-to-Node Transfer ---"
for i in "${!DEST_HOSTS[@]}"; do
  NODE_HOST=${DEST_HOSTS[$i]}
  NODE_NAME=${DEST_NAMES[$i]}

  echo "-----------------------------------------------"
  echo "🚀 Syncing otter1 -> $NODE_NAME"
  echo "-----------------------------------------------"

  # We use rsync -ahP:
  # -a: Archive mode (preserves permissions/links)
  # -h: Human readable numbers
  # -P: Shows progress bar and allows resuming partial transfers
  ssh -A "$SOURCE_NODE" "
        ssh -o StrictHostKeyChecking=no ${USER}@${NODE_HOST} 'mkdir -p $PARENT_DIR' && \
        rsync -ahP -e 'ssh -o StrictHostKeyChecking=no' $REMOTE_PATH/ ${USER}@${NODE_HOST}:$REMOTE_PATH/
    "

  if [ $? -eq 0 ]; then
    echo -e "\n✅ Done with $NODE_NAME\n"
  else
    echo -e "\n❌ Error syncing to $NODE_NAME\n"
  fi
done

echo "--- PHASE 2: Hardware Collection ---"
# (Hardware collection logic remains the same as before...)
ALL_NODES=("otter1" "otter2" "otter3" "otter4" "otter5" "otter6")
for NODE in "${ALL_NODES[@]}"; do
  echo "Querying $NODE..."
  {
    echo "NODE: $NODE"
    ssh "$NODE" "lscpu | grep -E 'Model name|CPU\(s\):'; echo '---'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU'"
    echo -e "-------------------------------------------\n"
  } >>"$REPORT_FILE"
done

echo "Script complete. Specs saved to $REPORT_FILE."
