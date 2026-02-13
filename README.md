# PedSense-AI 🚶‍♂️🔍
> **Predicting Pedestrian Crossing Intent through Multi-Stage Computer Vision**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Manager: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

PedSense is a high-performance computer vision framework designed to predict pedestrian crossing intent. By leveraging the **JAAD (Joint Attention in Autonomous Driving)** dataset, this project benchmarks three distinct architectural approaches to determine if a pedestrian is likely to cross the road or remain waiting at the curb.

---

## Project Architecture

PedSense evaluates three distinct models to identify the optimal balance between inference speed and behavioral accuracy:

1. **YOLOv26 (End-to-End Detector):** A single-stage pipeline fine-tuned to classify "Crossing" vs. "Waiting" directly within the object detection head. Optimized for maximum FPS.
2. **ResNet-50 + LSTM (Temporal Classifier):** A two-stage approach using ResNet for spatial feature extraction from pedestrian crops and an LSTM (Long Short-Term Memory) network to analyze movement patterns over multiple frames.
3. **The PedSense Hybrid (YOLO + ResNet):** A coordinated pipeline where YOLO acts as the **"Proposal Engine"** (finding and tracking pedestrians) and a dedicated ResNet-50 acts as the **"Decision Engine"** (providing high-resolution classification of the extracted crops).



---

## Tech Stack

* **Environment & Packaging:** [uv](https://github.com/astral-sh/uv) — 10-100x faster dependency resolution than pip.
* **Deep Learning:** [PyTorch](https://pytorch.org/) & [Ultralytics](https://github.com/ultralytics/ultralytics).
* **CLI Framework:** [Typer](https://typer.tiangolo.com/) & [Rich](https://github.com/Textualize/rich) for a professional, color-coded terminal experience.
* **UI/Demo:** [Gradio](https://gradio.app/) (Hosted on Hugging Face Spaces).
* **Model Registry:** [Hugging Face Hub](https://huggingface.co/) for versioned model weight storage.

---

## Quick Start

### 1. Installation
Ensure you have `uv` installed. If not, get it via `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
# Clone and enter the project
git clone [https://github.com/your-username/pedsense-ai.git](https://github.com/your-username/pedsense-ai.git)
cd pedsense-ai

# Synchronize the environment
uv sync

```

### 2. Usage

Manage the entire pipeline via the `pedsense` CLI entry point.

```bash
# Prepare and clean the JAAD dataset
uv run pedsense clean --input ./data/raw

# Train a model (yolo, resnet-lstm, or hybrid)
uv run pedsense train --model yolo

# Launch the Gradio web demo
uv run pedsense demo

```

---

## Licensing & Attribution

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

* **Models:** Utilizes Ultralytics YOLO (AGPL-3.0). Per license terms, this project's source code is fully public.
* **Dataset:** Based on the **JAAD Dataset** (MIT License).
* **Rationale:** AGPL-3.0 was selected to ensure that any derivative web services or demos remain open-source for the research community.

---

## Structure

```text
src/pedsense/
├── cli.py           # Typer CLI entry point
├── processing/      # JAAD parsing & dataset cleaning
├── models/          # YOLO, ResNet, and LSTM architectures
└── demo.py          # Gradio Web Interface logic

```

```