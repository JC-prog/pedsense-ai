# PedSense-AI

**Predicting Pedestrian Crossing Intent through Multi-Stage Computer Vision**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

PedSense-AI is a computer vision framework for predicting pedestrian crossing intent using the **JAAD (Joint Attention in Autonomous Driving)** dataset. It benchmarks three distinct architectural approaches to determine if a pedestrian is likely to cross the road.

---

## Architecture Overview

| Model | Approach | Strength |
|-------|----------|----------|
| **YOLO26** | End-to-end single-stage detector | Maximum FPS, real-time inference |
| **ResNet-50 + LSTM** | Two-stage temporal classifier | Temporal context from frame sequences |
| **Hybrid (YOLO + ResNet)** | YOLO proposals + ResNet classifier | Balanced speed and accuracy |

### YOLO26 (End-to-End)

Fine-tuned YOLO26 that detects pedestrians and classifies crossing intent (`crossing` / `not-crossing`) in a single forward pass. Optimized for real-time deployment.

### ResNet-50 + LSTM (Temporal)

ResNet-50 extracts spatial features from 16-frame sequences of pedestrian crops. An LSTM models temporal patterns (gait, posture changes) to predict crossing intent.

### Hybrid (YOLO + ResNet)

YOLO26 acts as the **Proposal Engine** (detecting pedestrians), and ResNet-50 acts as the **Decision Engine** (classifying crossing intent from cropped regions).

---

## Quick Links

- [Installation](getting-started/installation.md) — Get up and running
- [Quickstart](getting-started/quickstart.md) — End-to-end pipeline walkthrough
- [CLI Reference](cli/index.md) — All available commands
- [Model Architectures](models/index.md) — Detailed model documentation
- [Dataset Pipeline](dataset/index.md) — JAAD data processing
- [API Reference](api/index.md) — Module-level documentation
