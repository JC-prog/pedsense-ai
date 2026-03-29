# setup_windows.ps1 — Install pedsense-ai on Windows
# Supports two modes:
#   - CUDA (default): uses the pytorch-cu126 index via `uv sync`
#   - CPU only:       pass -CpuOnly to install standard PyPI wheels
#
# Usage:
#   .\scripts\setup_windows.ps1           # CUDA 12.6
#   .\scripts\setup_windows.ps1 -CpuOnly  # CPU only
param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

if ($CpuOnly) {
    Write-Host "==> Creating virtual environment..."
    uv venv

    Write-Host "==> Installing PyTorch (CPU) from PyPI..."
    uv pip install torch torchvision

    Write-Host "==> Installing remaining dependencies..."
    uv pip install `
        "typer>=0.15" `
        "rich>=13" `
        "opencv-python>=4.10" `
        "ultralytics>=8.3" `
        "gradio>=5.0" `
        "pandas>=2.2" `
        "huggingface-hub>=0.27"

    Write-Host "==> Installing pedsense package..."
    uv pip install -e .
} else {
    Write-Host "==> Creating virtual environment and syncing dependencies (CUDA 12.6)..."
    uv sync
}

Write-Host ""
Write-Host "Setup complete. Activate the environment with:"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "Then run: pedsense --help"
