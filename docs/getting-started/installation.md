# Installation

## Prerequisites

- **Python 3.12.10+** — [Download](https://www.python.org/downloads/)
- **uv** — Fast Python package manager. Install via:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **GPU (optional)** — CUDA-compatible GPU recommended for training. CPU works for preprocessing and small experiments.

## Clone and Install

```bash
git clone https://github.com/JCProg/pedsense-ai.git
cd pedsense-ai
uv sync
```

This installs all dependencies including PyTorch, Ultralytics, OpenCV, Gradio, and more.

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
data/raw/           # Place JAAD clips and annotations here
data/raw/frames/    # Extracted frames (generated)
data/processed/     # Formatted datasets (generated)
models/base/        # Downloaded pretrained weights
models/custom/      # Your trained models
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

To build documentation locally:

```bash
uv sync --group dev
uv run mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000).
