#!/bin/bash
# Install torch_geometric and its dependencies for GNN models
#
# torch_geometric requires specific versions matching your PyTorch installation.
# This script auto-detects your PyTorch version and installs compatible packages.

set -e

echo "==================================="
echo "Installing PyTorch Geometric (GNN)"
echo "==================================="

# Check if PyTorch is installed
python -c "import torch" 2>/dev/null || {
    echo "Error: PyTorch not found. Please install PyTorch first."
    exit 1
}

# Get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "PyTorch version: $TORCH_VERSION"

# Get Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Install torch_geometric
echo ""
echo "Installing torch_geometric..."
pip install torch_geometric

# Install optional dependencies for better performance
echo ""
echo "Installing optional dependencies (pyg_lib, torch_sparse, torch_scatter)..."
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html 2>/dev/null || {
    echo "Warning: Could not install pre-built wheels for your PyTorch version."
    echo "Installing from source (this may take a while)..."
    pip install torch_scatter torch_sparse
}

echo ""
echo "==================================="
echo "Installation complete!"
echo "==================================="
echo ""
echo "Verify installation:"
python -c "from torch_geometric.nn import GCNConv; print('✓ torch_geometric installed successfully')"
