# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PedSense-AI is a computer vision framework for predicting pedestrian crossing intent using the JAAD (Joint Attention in Autonomous Driving) dataset. It implements three model architectures:

1. **YOLO26** — End-to-end 2-class detector (`crossing` / `not-crossing`), fine-tuned from `yolo26n.pt`
2. **ResNet-50 + LSTM** — Temporal classifier over 16-frame pedestrian crop sequences
3. **Hybrid (YOLO + ResNet)** — YOLO26 1-class pedestrian detector + ResNet-50 single-frame intent classifier

## Development Commands

```bash
# Install/sync dependencies (includes CUDA PyTorch via configured index)
uv sync

# Run the CLI
uv run pedsense setup                                      # Create project directories
uv run pedsense preprocess frames                           # Extract frames from JAAD videos
uv run pedsense preprocess yolo                             # Convert to YOLO format
uv run pedsense preprocess resnet                           # Convert to ResNet+LSTM sequences
uv run pedsense preprocess all                              # Run full preprocessing pipeline
uv run pedsense train -m yolo -n my_yolo -e 50 -b 16       # Train YOLO26
uv run pedsense train -m resnet-lstm -n my_resnet -e 30     # Train ResNet+LSTM
uv run pedsense train -m hybrid -n my_hybrid                # Train Hybrid pipeline
uv run pedsense demo                                        # Launch Gradio demo
```

No test suite or linter is configured yet.

## Architecture

- **Package manager:** uv (with `uv_build` backend)
- **Python:** >=3.12.10
- **CLI:** Typer + Rich (`src/pedsense/cli.py` → entry point `pedsense`)
- **Source layout:** `src/pedsense/` (src-layout pattern)
- **GPU:** PyTorch CUDA 12.6 via `[[tool.uv.index]]` in `pyproject.toml`

### Module Layout

```
src/pedsense/
    cli.py                  # Typer CLI (setup, preprocess, train, demo)
    config.py               # Path constants & training defaults
    demo.py                 # Gradio web interface for inference
    processing/
        annotations.py      # CVAT XML parser → BoundingBox, Track, VideoAnnotation
        frames.py            # OpenCV frame extraction from MP4
        yolo_format.py       # YOLO dataset converter (2-class)
        resnet_format.py     # ResNet+LSTM sequence converter (16-frame crops)
    train/
        yolo_trainer.py      # YOLO26 fine-tuning via Ultralytics
        resnet_lstm.py       # ResNetLSTM & ResNetClassifier (nn.Module)
        resnet_trainer.py    # ResNet+LSTM training loop
        hybrid_trainer.py    # Hybrid 2-stage pipeline trainer
```

### Key Configuration

- Base pretrained models download to `models/base/` (not project root)
- Trained models save to `models/custom/{name}_{YYYYMMDD_HHMMSS}/`
- JAAD raw data expected in `data/raw/clips/` and `data/raw/annotations/`
- Processed datasets in `data/processed/yolo/` and `data/processed/resnet/`

The CLI app is registered as a script entry point in `pyproject.toml`:
```
pedsense = "pedsense.cli:app"
```

## License

AGPL-3.0 — intentionally chosen for research community sharing.
