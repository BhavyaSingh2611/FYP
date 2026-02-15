#!/usr/bin/env bash
# Fetch /scratch/bs01346/FYP/runs from all 6 otter machines to local disk.
# Organises downloads under /Users/bhavya/Documents/fyp_runs/DDMMYYYY/<host>/

set -euo pipefail

REMOTE_DIR="/scratch/bs01346/FYP/runs"
LOCAL_BASE="/Users/bhavya/Documents/fyp_runs"
DATE_DIR=$(date +"%d%m%Y")
DEST="${LOCAL_BASE}/${DATE_DIR}"

mkdir -p "$DEST"

for host in otter{1..6}; do
  echo "[$host] Fetching runs..."
  mkdir -p "${DEST}/${host}"
  rsync -avz --include='*/' --include='*.pt' --exclude='*' -e "ssh -o StrictHostKeyChecking=no" "${host}:${REMOTE_DIR}/" "${DEST}/${host}/" &
done

echo "Waiting for all transfers to finish..."
wait

echo ""
echo "All runs saved to: ${DEST}"
ls -1 "$DEST"
