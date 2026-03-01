# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] - v2.0.0

### Added

- `pedsense export` CLI command to convert trained models to ONNX format
- Standalone demo app using `onnxruntime` (no PyTorch dependency required)
- Lighter deployment footprint for inference-only environments

## [1.2.0] - 2026-03-01

### Added

- `--track-labels / -t` option for `pedsense preprocess` — select which track label types to include (repeatable: `-t pedestrian -t traffic_light`)
- Multi-class YOLO detection: non-pedestrian tracks (`traffic_light`, `crosswalk`) are appended as additional YOLO classes after the behavioral attribute classes
- `PEDESTRIAN_LABELS` constant in `pedsense.processing.annotations` — the three pedestrian-variant track labels (`pedestrian`, `ped`, `people`) that support behavioral attribute classification
- `TRACK_LABELS` constant listing all known JAAD track label types
- `pedsense attributes` now shows both behavioral attributes and track label types
- `convert_to_resnet()` accepts `ped_labels` to include `ped` and `people` variants in ResNet+LSTM sequences

## [1.1.0] - 2026-03-01

### Added

- `pedsense attributes` CLI command — lists all supported JAAD behavioral attributes and their class values
- `--attribute / -a` option for `pedsense preprocess` — choose which annotation attribute to classify on (`cross`, `action`, `look`, `occlusion`); default remains `cross`
- `--fps` option for `pedsense preprocess frames` — downsample frame extraction to a target FPS while preserving original frame indices for annotation compatibility
- `ATTRIBUTE_LABELS` constant in `pedsense.processing.annotations` — single source of truth for attribute names and class orderings across all preprocessors and the trainer
- ResNet+LSTM trainer now accepts an `attribute` parameter; `num_classes`, label map, and saved `config.json` are all derived dynamically

## [1.0.2] - 2026-02-15

### Added

- Unit test suite with pytest (`tests/test_demo_helpers.py`, `tests/test_annotations.py`)
- pytest added to dev dependencies

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
