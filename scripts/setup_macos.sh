#!/usr/bin/env bash
# setup_macos.sh — Install pedsense-ai on macOS (CPU / Apple Silicon MPS)
# The main pyproject.toml pins torch to the CUDA 12.6 index (Linux/Windows only).
# This script installs the standard PyPI builds (CPU + MPS support) instead.
set -euo pipefail

echo "==> Creating virtual environment..."
uv venv

echo "==> Installing PyTorch (CPU/MPS) from PyPI..."
uv pip install torch torchvision

echo "==> Installing remaining dependencies..."
uv pip install \
    "typer>=0.15" \
    "rich>=13" \
    "opencv-python>=4.10" \
    "ultralytics>=8.3" \
    "gradio>=5.0" \
    "pandas>=2.2" \
    "huggingface-hub>=0.27"

echo "==> Installing pedsense package..."
uv pip install -e .

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo "Then run: pedsense --help"
