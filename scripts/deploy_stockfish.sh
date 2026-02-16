#!/usr/bin/env bash
# Deploy Stockfish binary to all 6 otter VMs.
# No sudo required — installs to /scratch/bs01346/FYP/stockfish/

set -euo pipefail

LOCAL_TAR="stockfish/stockfish-ubuntu-x86-64-avxvnni.tar"
REMOTE_DIR="/scratch/bs01346/FYP/stockfish"

if [[ ! -f "$LOCAL_TAR" ]]; then
  echo "ERROR: $LOCAL_TAR not found"
  exit 1
fi

for host in otter{1..6}; do
  echo "[$host] Creating $REMOTE_DIR ..."
  ssh -o StrictHostKeyChecking=no "$host" "mkdir -p ${REMOTE_DIR}"

  echo "[$host] Uploading $(basename "$LOCAL_TAR") ..."
  scp -o StrictHostKeyChecking=no "$LOCAL_TAR" "${host}:${REMOTE_DIR}/"

  echo "[$host] Extracting ..."
  ssh -o StrictHostKeyChecking=no "$host" \
    "cd ${REMOTE_DIR} && tar xf $(basename "$LOCAL_TAR") && chmod +x stockfish/stockfish-ubuntu-x86-64-avxvnni"

  echo "[$host] Done."
  echo ""
done

echo "Stockfish deployed to all 6 otter VMs at ${REMOTE_DIR}/stockfish/stockfish-ubuntu-x86-64-avxvnni"
