#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing package in editable mode..."
pip install -e ".[dev]"

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
