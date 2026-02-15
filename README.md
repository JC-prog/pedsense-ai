# PedSense-AI
> **Predicting Pedestrian Crossing Intent through Multi-Stage Computer Vision**

[![Version: 1.0.1](https://img.shields.io/badge/version-1.0.1-green.svg)](CHANGELOG.md)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Manager: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://jcprog.github.io/pedsense-ai/)

PedSense-AI is a computer vision framework for predicting pedestrian crossing intent using the **JAAD (Joint Attention in Autonomous Driving)** dataset. It benchmarks three distinct architectural approaches to determine whether a pedestrian is likely to cross the road or remain waiting at the curb.

---

## Model Architectures

| Model | Approach | Temporal | Best For |
|-------|----------|----------|----------|
| **YOLO26** | End-to-end single-stage detector | No | Real-time inference, maximum FPS |
| **ResNet-50 + LSTM** | Two-stage temporal classifier | Yes (16 frames) | Accuracy-critical scenarios |
| **Hybrid (YOLO + ResNet)** | YOLO proposals + ResNet classifier | No | Balanced speed and accuracy |

1. **YOLO26 (End-to-End Detector):** Fine-tuned to classify `crossing` vs. `not-crossing` directly within the detection head. Two classes, single forward pass per frame.
2. **ResNet-50 + LSTM (Temporal Classifier):** ResNet-50 extracts spatial features from 16-frame sequences of pedestrian crops. An LSTM models temporal patterns — gait, posture changes — to predict crossing intent.
3. **Hybrid (YOLO + ResNet):** YOLO26 acts as the **Proposal Engine** (1-class pedestrian detection) and ResNet-50 acts as the **Decision Engine** (single-frame crossing intent classification on each crop).

---

## Tech Stack

| Category | Tool |
|----------|------|
| Environment & Packaging | [uv](https://github.com/astral-sh/uv) |
| Deep Learning | [PyTorch](https://pytorch.org/) & [Ultralytics](https://github.com/ultralytics/ultralytics) |
| CLI | [Typer](https://typer.tiangolo.com/) & [Rich](https://github.com/Textualize/rich) |
| Demo UI | [Gradio](https://gradio.app/) |
| Model Registry | [Hugging Face Hub](https://huggingface.co/) |
| Documentation | [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) |

---

## Quick Start

### 1. Install

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install (includes CUDA PyTorch for GPU training)
git clone https://github.com/JCProg/pedsense-ai.git
cd pedsense-ai
uv sync
```

### 2. Download JAAD Dataset

1. Download video clips from the [JAAD dataset page](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) → `data/raw/clips/`
2. Download annotations from [JAAD GitHub](https://github.com/ykotseruba/JAAD) → `data/raw/annotations/`

```
data/raw/
├── clips/
│   ├── video_0001.mp4 ... video_0346.mp4
└── annotations/
    ├── video_0001.xml ... video_0346.xml
```

### 3. Prepare Data

```bash
# Create project directories
uv run pedsense setup

# Extract frames from all videos
uv run pedsense preprocess frames

# Convert to YOLO format (2-class detection)
uv run pedsense preprocess yolo

# Convert to ResNet+LSTM format (16-frame sequences)
uv run pedsense preprocess resnet

# Or run the full pipeline at once
uv run pedsense preprocess all
```

### 4. Train

```bash
# YOLO26 — end-to-end crossing intent detection
uv run pedsense train -m yolo -n my_yolo -e 50 -b 16

# ResNet+LSTM — temporal sequence classifier
uv run pedsense train -m resnet-lstm -n my_resnet -e 30 -b 8

# Hybrid — YOLO detector + ResNet classifier
uv run pedsense train -m hybrid -n my_hybrid -e 30
```

Models are saved to `models/custom/{name}_{YYYYMMDD_HHMMSS}/`.

### 5. Demo

```bash
# Launch Gradio with the most recent model
uv run pedsense demo

# Or specify a trained model
uv run pedsense demo -m my_yolo_20260214_153000
```

Open [http://localhost:7860](http://localhost:7860) — upload a video to see crossing intent predictions with annotated bounding boxes.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pedsense setup` | Create project directory structure |
| `pedsense preprocess [STEP]` | Extract frames and prepare datasets (`frames`, `yolo`, `resnet`, or `all`) |
| `pedsense train -m MODEL` | Train a model (`yolo`, `resnet-lstm`, or `hybrid`) |
| `pedsense demo` | Launch Gradio web interface |

### Key Options

```bash
# Preprocess a single video
uv run pedsense preprocess frames -v video_0001

# Train with custom name, epochs, batch size
uv run pedsense train -m yolo -n experiment1 -e 100 -b 32

# Hybrid: reuse an existing YOLO detector
uv run pedsense train -m hybrid --yolo-model models/custom/my_yolo/weights/best.pt

# Demo on a custom port
uv run pedsense demo -p 8080
```

Full CLI docs: [CLI Reference](https://jcprog.github.io/pedsense-ai/cli/)

---

## Project Structure

```text
pedsense-ai/
├── src/pedsense/
│   ├── cli.py                 # Typer CLI entry point
│   ├── config.py              # Path constants & training defaults
│   ├── demo.py                # Gradio web interface
│   ├── processing/
│   │   ├── annotations.py     # CVAT XML annotation parser
│   │   ├── frames.py          # Video frame extraction (OpenCV)
│   │   ├── yolo_format.py     # YOLO dataset converter
│   │   └── resnet_format.py   # ResNet+LSTM sequence converter
│   └── train/
│       ├── yolo_trainer.py    # YOLO26 fine-tuning (Ultralytics)
│       ├── resnet_lstm.py     # ResNetLSTM & ResNetClassifier (nn.Module)
│       ├── resnet_trainer.py  # ResNet+LSTM training loop
│       └── hybrid_trainer.py  # Hybrid 2-stage pipeline trainer
├── data/
│   ├── raw/                   # JAAD clips, annotations, extracted frames
│   └── processed/             # YOLO and ResNet formatted datasets
├── models/
│   ├── base/                  # Downloaded pretrained weights
│   └── custom/                # Trained model outputs
├── docs/                      # MkDocs documentation source
├── mkdocs.yml                 # MkDocs configuration
├── CHANGELOG.md               # Version history
└── pyproject.toml
```

---

## Documentation

Full documentation is available at [jcprog.github.io/pedsense-ai](https://jcprog.github.io/pedsense-ai/).

To build docs locally:

```bash
uv sync --group dev
uv run mkdocs serve    # Preview at localhost:8000
```

---

## License

**AGPL-3.0** — intentionally chosen for research community sharing.

* **Models:** Ultralytics YOLO (AGPL-3.0) — source code is fully public per license terms.
* **Dataset:** JAAD Dataset (MIT License).
* **Rationale:** AGPL-3.0 ensures any derivative web services or demos remain open-source.
