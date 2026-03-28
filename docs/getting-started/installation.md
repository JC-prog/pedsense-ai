# Installation

## Prerequisites

- **Python 3.12.10+** — [Download](https://www.python.org/downloads/)
- **uv** — Fast Python package manager. Install via:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **GPU (recommended)** — CUDA-compatible NVIDIA GPU for training. CPU works for preprocessing and small experiments.

## Clone the Repository

```bash
git clone https://github.com/JCProg/pedsense-ai.git
cd pedsense-ai
```

## Platform-Specific Installation

PyTorch wheels differ by platform. Use the script that matches your OS.

### Linux (CUDA 12.6)

```bash
bash scripts/setup_linux_cuda.sh
```

This runs `uv sync`, which pulls `torch` and `torchvision` from the PyTorch CUDA 12.6 wheel index automatically.

### macOS (CPU / Apple Silicon MPS)

macOS has no CUDA support. The setup script installs the standard PyPI builds, which include MPS (Metal Performance Shaders) acceleration on Apple Silicon.

```bash
bash scripts/setup_macos.sh
source .venv/bin/activate
```

### Windows

**With CUDA 12.6 (recommended for NVIDIA GPUs):**

```powershell
.\scripts\setup_windows.ps1
.venv\Scripts\Activate.ps1
```

**CPU only:**

```powershell
.\scripts\setup_windows.ps1 -CpuOnly
.venv\Scripts\Activate.ps1
```

## Verify GPU Access

After installation, confirm PyTorch can see your hardware:

```bash
# NVIDIA GPU (Linux / Windows)
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Apple Silicon (macOS)
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

!!! note
    If CUDA shows `False` on Linux/Windows, ensure your NVIDIA drivers are up to date and compatible with CUDA 12.6.

## Verify Installation

```bash
uv run pedsense --help
```

You should see:

```
Usage: pedsense [OPTIONS] COMMAND [ARGS]...

 PedSense: Pedestrian Intent Prediction Suite

Commands:
  setup       Verify project structure and prepare environment.
  preprocess  Extract frames and prepare datasets from raw JAAD data.
  train       Train a model for pedestrian crossing intent prediction.
  demo        Launch the Gradio web interface for inference.
```

## Set Up Project Structure

```bash
uv run pedsense setup
```

This creates all required directories:

```
data/raw/               # Place JAAD clips and annotations here
data/raw/frames/        # Extracted frames (generated)
data/processed/yolo/    # YOLO-formatted dataset (generated)
data/processed/resnet/  # ResNet+LSTM sequences (generated)
models/base/            # Downloaded pretrained weights
models/detector/        # Detection models (yolo, yolo-pose, hybrid)
models/classifier/      # Intent classifiers (keypoint-lstm, resnet-lstm)
```

## Download JAAD Dataset

1. Visit the [JAAD dataset page](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
2. Download the video clips and place them in `data/raw/clips/`
3. Download the annotations from [GitHub](https://github.com/ykotseruba/JAAD) and place them in `data/raw/annotations/`

Expected structure:

```
data/raw/
    clips/
        video_0001.mp4
        video_0002.mp4
        ...
        video_0346.mp4
    annotations/
        video_0001.xml
        video_0002.xml
        ...
        video_0346.xml
```

## Development Setup

Install dev dependencies (pytest, mkdocs):

```bash
uv sync --group dev
```

### Run Tests

```bash
uv run pytest tests/ -v
```

Tests cover demo model discovery helpers and CVAT XML annotation parsing. No GPU or dataset required.

### Build Documentation

```bash
uv run mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000).
