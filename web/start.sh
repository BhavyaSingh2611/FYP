#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Chess ML Arena ==="
echo ""

# Install Flask
echo "→ Installing Flask..."
.venv/bin/pip install flask -q

# Build frontend
echo "→ Building frontend..."
cd web/client
npm install --silent
npm run build
cd ../..

echo ""
echo "→ Starting server..."
echo ""
.venv/bin/python web/server.py
