# PedSense-AI
> **Predicting Pedestrian Crossing Intent through Multi-Stage Computer Vision**

[![Version: 1.6.5](https://img.shields.io/badge/version-1.6.5-green.svg)](CHANGELOG.md)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Manager: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://jcprog.github.io/pedsense-ai/)

PedSense-AI is a computer vision framework for predicting pedestrian crossing intent using the **JAAD (Joint Attention in Autonomous Driving)** dataset. It benchmarks three distinct architectural approaches to determine whether a pedestrian is likely to cross the road or remain waiting at the curb.

The core upstream pipeline runs YOLO-Pose on JAAD frames, aligns detections to annotated pedestrian tracks via IoU matching, and builds normalized keypoint sequences `(T, 17, 2)` anchored to each pedestrian's `crossing_point` - ready for temporal models (LSTM, ST-GCN).

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
| Testing | [pytest](https://docs.pytest.org/) |
| Model Registry | [Hugging Face Hub](https://huggingface.co/) |
| Documentation | [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) |

---

## Pretrained Models

Want to try the demo without training? Download the best models with a single command.

| Role | Model | HF Repo |
|------|-------|---------|
| Pose Detector | YOLO26m-Pose · JAAD · 10 epochs | [JcProg/pedsense-yolo26m-pose-jaad-10e](https://huggingface.co/JcProg/pedsense-yolo26m-pose-jaad-10e) |
| Intent Classifier | KeypointLSTM · JAAD · 50 epochs | [JcProg/pedsense-keypoint-lstm-jaad-50e](https://huggingface.co/JcProg/pedsense-keypoint-lstm-jaad-50e) |

```bash
uv run pedsense download JcProg/pedsense-yolo26m-pose-jaad-10e
uv run pedsense download JcProg/pedsense-keypoint-lstm-jaad-50e
```

Then launch the demo and select the **2-Stage Intent (Pose + LSTM)** pipeline:

```bash
uv run pedsense demo
```

---

## Quick Start

### 1. Install

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/JCProg/pedsense-ai.git
cd pedsense-ai
```

Use the setup script for your platform:

| Platform | Command |
|----------|---------|
| **Linux** (CUDA 12.6) | `bash scripts/setup_linux_cuda.sh` |
| **macOS** (CPU / MPS) | `bash scripts/setup_macos.sh` |
| **Windows** (CUDA 12.6) | `.\scripts\setup_windows.ps1` |
| **Windows** (CPU only) | `.\scripts\setup_windows.ps1 -CpuOnly` |

> macOS has no CUDA support. The macOS script installs standard PyPI wheels with MPS acceleration for Apple Silicon.

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

# Build keypoint sequence dataset (YOLO-Pose → JAAD alignment → (T,17,2) arrays)
uv run pedsense preprocess keypoints

# Or run frames + yolo + resnet at once
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

Detection models save to `models/detector/`, intent classifiers to `models/classifier/`.

### 5. Demo

```bash
# Launch Gradio with the most recent model
uv run pedsense demo

# Or pre-select a model
uv run pedsense demo -m my_yolo_20260214_153000
```

Open [http://localhost:7860](http://localhost:7860). Two pipelines are available:

- **Detection Only** — frame-by-frame inference with YOLO, YOLO-Pose (skeleton), or Hybrid
- **2-Stage Intent (Pose + LSTM)** — select a YOLO-Pose model and a trained KeypointLSTM model; pedestrians are tracked across frames, keypoints buffered per pedestrian, and intent classified once T frames are collected

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pedsense setup` | Create project directory structure |
| `pedsense download <repo_id>` | Download a model from Hugging Face Hub into the correct local directory |
| `pedsense preprocess [STEP]` | Extract frames and prepare datasets (`frames`, `yolo`, `resnet`, `pose`, `keypoints`, or `all`) |
| `pedsense train -m MODEL` | Train a model (`yolo`, `yolo-detector`, `yolo-pose`, `resnet-lstm`, or `hybrid`) |
| `pedsense resume` | Interactively resume a YOLO training run |
| `pedsense demo` | Launch Gradio web interface |
| `pedsense attributes` | List annotation attributes and track label types |

### Key Options

```bash
# Preprocess a single video
uv run pedsense preprocess frames -v video_0001

# Build keypoint dataset with a larger pose model and stricter IoU
uv run pedsense preprocess keypoints --pose-variant yolo11s-pose --iou-threshold 0.4

# Override sequence window length and stride
uv run pedsense preprocess keypoints --sequence-length 30 --sequence-stride 10

# Default 1s horizon — FPS read from meta.json written during frame extraction
uv run pedsense preprocess keypoints

# Override to a longer horizon
uv run pedsense preprocess keypoints --prediction-horizon 2.0

# Train with custom name, epochs, batch size
uv run pedsense train -m yolo -n experiment1 -e 100 -b 32

# Hybrid: reuse an existing YOLO detector
uv run pedsense train -m hybrid --yolo-model models/detector/my_yolo/weights/best.pt

# Demo on a custom port
uv run pedsense demo -p 8080
```

Full CLI docs: [CLI Reference](https://jcprog.github.io/pedsense-ai/cli/)

---

## Project Structure

```text
pedsense-ai/
├── src/pedsense/
│   ├── cli.py                      # Typer CLI entry point
│   ├── config.py                   # Path constants & training defaults
│   ├── demo.py                     # Gradio web interface
│   ├── processing/
│   │   ├── annotations.py          # CVAT XML annotation parser
│   │   ├── frames.py               # Video frame extraction (OpenCV)
│   │   ├── yolo_format.py          # YOLO dataset converter
│   │   ├── resnet_format.py        # ResNet+LSTM sequence converter
│   │   ├── pose_format.py          # YOLO-Pose label extractor
│   │   └── keypoint_pipeline.py   # YOLO-Pose → JAAD alignment → (T,17,2) sequences
│   └── train/
│       ├── yolo_trainer.py         # YOLO26 & YOLO-Pose fine-tuning (Ultralytics)
│       ├── resnet_lstm.py          # ResNetLSTM & ResNetClassifier (nn.Module)
│       ├── resnet_trainer.py       # ResNet+LSTM training loop
│       └── hybrid_trainer.py       # Hybrid 2-stage pipeline trainer
├── data/
│   ├── raw/                        # JAAD clips, annotations, extracted frames
│   └── processed/
│       ├── yolo/                   # YOLO detection dataset
│       ├── resnet/                 # ResNet+LSTM crop sequences
│       ├── pose/                   # YOLO-Pose label files
│       └── keypoints/              # (T,17,2) keypoint sequences + labels.csv
├── models/
│   ├── base/                       # Downloaded pretrained weights
│   ├── detector/                   # Detection models (yolo, yolo-pose, hybrid)
│   └── classifier/                 # Intent classifiers (keypoint-lstm, resnet-lstm)
├── tests/
│   ├── test_demo_helpers.py        # Demo utility function tests
│   └── test_annotations.py        # XML annotation parser tests
├── docs/                           # MkDocs documentation source
├── mkdocs.yml                      # MkDocs configuration
├── CHANGELOG.md                    # Version history
└── pyproject.toml
```

---

## Testing

```bash
uv sync --group dev
uv run pytest tests/ -v
```

Tests cover demo model discovery helpers and CVAT XML annotation parsing. No GPU or dataset required.

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
