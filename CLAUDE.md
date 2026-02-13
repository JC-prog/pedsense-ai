# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PedSense-AI is a computer vision framework for predicting pedestrian crossing intent using the JAAD (Joint Attention in Autonomous Driving) dataset. It evaluates three model architectures:

1. **YOLOv26** — End-to-end single-stage detector optimized for real-time FPS
2. **ResNet-50 + LSTM** — Two-stage temporal classifier (spatial features → sequence modeling)
3. **Hybrid (YOLO + ResNet)** — YOLO proposal engine feeding a ResNet decision engine

The project is early-stage (v0.1.0) — CLI scaffolding exists but core ML functionality is not yet implemented.

## Development Commands

```bash
# Install/sync dependencies
uv sync

# Run the CLI
uv run pedsense setup                          # Create project folders (data/raw, data/processed, models)
uv run pedsense train --model <yolo|resnet-lstm|hybrid>  # Train a model (placeholder)
uv run pedsense demo                            # Launch Gradio demo (placeholder)
```

No test suite or linter is configured yet.

## Architecture

- **Package manager:** uv (with `uv_build` backend)
- **Python:** >=3.12.10
- **CLI:** Typer + Rich (`src/pedsense/cli.py` → entry point `pedsense`)
- **Source layout:** `src/pedsense/` (src-layout pattern)

The CLI app is registered as a script entry point in `pyproject.toml`:
```
pedsense = "pedsense.cli:app"
```

Planned but not yet implemented modules: dataset processing, model architectures (PyTorch/Ultralytics), Gradio demo interface, Hugging Face Hub integration.

## License

AGPL-3.0 — intentionally chosen for research community sharing.
