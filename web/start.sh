#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Virtual environment not found. Run setup.sh first."
  exit 1
fi

echo "Building frontend"

cd "$ROOT_DIR/web/client"
npm install --silent
npm run build
cd "$ROOT_DIR"

echo "Starting server"
"$VENV_PY" web/server.py