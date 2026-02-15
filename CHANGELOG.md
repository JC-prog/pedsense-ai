# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - v2.0.0

### Added

- `pedsense export` CLI command to convert trained models to ONNX format
- Standalone demo app using `onnxruntime` (no PyTorch dependency required)
- Lighter deployment footprint for inference-only environments

## [1.0.1] - 2026-02-15

### Fixed

- Demo model dropdown not showing YOLO models when weights were renamed or `best.pt` was missing
- Model discovery now uses fallback priority: `best.pt` > `last.pt` > any `.pt` file

## [1.0.0] - 2026-02-15

### Added

- YOLO26 end-to-end 2-class detector (`crossing` / `not-crossing`), fine-tuned from `yolo26n.pt`
- ResNet-50 + LSTM temporal classifier over 16-frame pedestrian crop sequences
- Hybrid pipeline: YOLO26 1-class pedestrian detector + ResNet-50 single-frame intent classifier
- CLI tool (`pedsense`) with commands: `setup`, `preprocess`, `train`, `demo`
- Preprocessing pipeline: frame extraction, YOLO format conversion, ResNet sequence generation
- Gradio web interface for model inference on video
- JAAD (Joint Attention in Autonomous Driving) dataset support
- Class-weighted loss for handling imbalanced crossing/not-crossing labels
- Cosine annealing learning rate scheduler for ResNet+LSTM training
- Automatic model discovery and selection in the demo interface
