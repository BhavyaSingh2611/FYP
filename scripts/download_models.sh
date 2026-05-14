#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://pub-d7b724eb8d2f4662ae366f55398d2b04.r2.dev"
REMOTE_ROOT="models"
LOCAL_ROOT="runs/models"

models=(
  "convnet/convnet_10M_e50.pt"
  "convnet/convnet_100M_e15.pt"
  "convnet/convnet_200M_e10.pt"
  "convnet/convnet_500M_e5.pt"
  "convnet/convnet_1000M_e1.pt"
  "gat/gat_10M_e50.pt"
  "gat/gat_100M_e15.pt"
  "gcn/gcn_10M_e50.pt"
  "gcn/gcn_100M_e15.pt"
  "piece_transformer/piece_transformer_10M_e50.pt"
  "piece_transformer/piece_transformer_100M_e15.pt"
  "piece_transformer/piece_transformer_200M_e10.pt"
  "piece_transformer/piece_transformer_500M_e5.pt"
  "piece_transformer/piece_transformer_1000M_e1.pt"
  "resnet/resnet_10M_e50.pt"
  "resnet/resnet_100M_e15.pt"
  "resnet/resnet_200M_e10.pt"
  "resnet/resnet_500M_e5.pt"
  "resnet/resnet_1000M_e1.pt"
  "square_transformer/square_transformer_10M_e50.pt"
  "square_transformer/square_transformer_100M_e15.pt"
  "square_transformer/square_transformer_200M_e10.pt"
  "square_transformer/square_transformer_500M_e5.pt"
  "square_transformer/square_transformer_1000M_e1.pt"
)

mkdir -p "${LOCAL_ROOT}"

for rel_path in "${models[@]}"; do
  dest_path="${LOCAL_ROOT}/${rel_path}"

  mkdir -p "$(dirname "${dest_path}")"
  curl -fL "${BASE_URL}/${REMOTE_ROOT}/${rel_path}" -o "${dest_path}"
  
  echo "Downloaded: ${dest_path}"
done