#!/usr/bin/env bash
# setup_linux_cuda.sh — Install pedsense-ai on Linux with CUDA 12.6
# Uses the standard `uv sync` flow — pyproject.toml already configures
# the pytorch-cu126 index for torch and torchvision.
set -euo pipefail

echo "==> Creating virtual environment and syncing dependencies (CUDA 12.6)..."
uv sync

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo "Then run: pedsense --help"
