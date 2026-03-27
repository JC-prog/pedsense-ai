# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.6.5] - 2026-03-28

### Added

- `results.csv` written to the KeypointLSTM model output directory after each training run — records `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_f1`, `val_auc` per epoch, consistent with YOLO trainer output

## [1.6.4] - 2026-03-28

### Fixed

- 2-Stage Intent demo showed `buffering 0/T` indefinitely when a non-pose model was selected as the Pose Detector: per-track buffers were created for every bounding box regardless of whether keypoints were present, so the buffer existed but was never filled
- Per-track buffers now created only on the first frame where valid keypoints are received for that track; pedestrians with no keypoint output show `"no keypoints"` instead of a frozen buffer counter
- Added early validation: if `r.keypoints is None` on the first detected frame, inference stops with a clear error directing the user to select a YOLO-Pose model

### Added

- `_resolve_model_dir(name)` helper searches `models/detector/` then `models/classifier/` — `run_inference` resolves model paths from either directory without hardcoding

## [1.6.3] - 2026-03-27

### Changed

- `models/custom/` split into `models/detector/` (yolo, yolo-pose, hybrid) and `models/classifier/` (keypoint-lstm, resnet-lstm)
- `CUSTOM_MODELS_DIR` constant replaced by `DETECTOR_MODELS_DIR` and `CLASSIFIER_MODELS_DIR` in `pedsense.config`
- `pedsense setup` now creates both `models/detector/` and `models/classifier/`
- `pedsense resume` scans `models/detector/` instead of `models/custom/`

## [1.6.2] - 2026-03-27

### Added

- **2-Stage Intent pipeline in demo** — new pipeline selector in the Gradio UI lets you pair a YOLO-Pose pose detector with a trained KeypointLSTM model for online crossing intent classification
- `_run_keypoint_lstm_inference()` — online inference runner using YOLO byte-tracker for stable per-pedestrian IDs; per-track deques buffer T normalized keypoint frames before triggering LSTM classification; bounding boxes shown in yellow while buffering, green/red once classified
- `_list_models_by_type()` — scans `models/detector/` and `models/classifier/` and groups results into `detection` (yolo/yolo-pose/hybrid) and `keypoint-lstm` buckets for UI filtering
- `_load_keypoint_lstm_model()` — loads `KeypointLSTM` weights and reads `input_size` and `sequence_length` from `config.json`
- `sequence_length` field written to `config.json` after `train -m keypoint-lstm` — detected from the first training sample so the demo can size frame buffers without extra arguments

### Changed

- `run_inference` signature extended with `pipeline` and `intent_model_name` parameters; routes to the appropriate runner based on pipeline selection
- Demo UI now shows two model dropdowns in 2-Stage mode (pose detector + intent model); labels and choices update dynamically when the pipeline radio changes
- `keypoint-lstm` model type now returns a useful error message in Detection Only mode, directing the user to switch to the 2-Stage pipeline

## [Unreleased] - v2.0.0

### Added

- `pedsense export` CLI command to convert trained models to ONNX format
- Standalone demo app using `onnxruntime` (no PyTorch dependency required)
- Lighter deployment footprint for inference-only environments

## [1.5.0] - 2026-03-07

### Added

- `train -m yolo-pose` CLI command — fine-tune a YOLO-Pose model on `data/processed/pose/`
- `--yolo-variant` accepts YOLO-Pose variants for `yolo-pose`: `yolo11n-pose` (default), `yolo11s-pose`, `yolo11m-pose`; consistent with other YOLO commands defaulting to nano
- `train_yolo_pose()` function in `pedsense.train.yolo_trainer`

## [1.4.0] - 2026-03-07

### Added

- `preprocess pose` step — runs a pretrained YOLO-Pose model on extracted frames to generate YOLO pose-format labels (bounding box + 17 COCO keypoints per pedestrian)
- `--pose-variant TEXT` option for `pedsense preprocess` — select YOLO-Pose model: `yolo11n-pose` (default), `yolo11s-pose`, `yolo11m-pose`; applies to `pose` step
- `--conf FLOAT` option for `pedsense preprocess` — detection confidence threshold for pose extraction (default: `0.25`); applies to `pose` step
- `extract_pose_labels()` function in `pedsense.processing.pose_format`
- `POSE_DIR` path constant in `pedsense.config`
- `data/processed/pose/data.yaml` includes `kpt_shape: [17, 3]` for direct use with Ultralytics YOLO-Pose training

## [1.3.2] - 2026-03-04

### Added

- `--aug-degrees FLOAT` option for `pedsense train` — rotation augmentation range in degrees (default: `0.0`); applies to `yolo`, `yolo-detector`
- `--aug-scale FLOAT` option for `pedsense train` — scale jitter fraction (default: `0.5`); applies to `yolo`, `yolo-detector`
- `--aug-mosaic FLOAT` option for `pedsense train` — mosaic augmentation probability (default: `1.0`); applies to `yolo`, `yolo-detector`
- `--aug-mixup FLOAT` option for `pedsense train` — mixup augmentation probability (default: `0.0`); applies to `yolo`, `yolo-detector`
- `--aug-fliplr FLOAT` option for `pedsense train` — horizontal flip probability (default: `0.5`); applies to `yolo`, `yolo-detector`
- `degrees`, `scale`, `mosaic`, `mixup`, `fliplr` parameters added to `train_yolo()` and `train_yolo_detector()` API functions

## [1.3.1] - 2026-03-03

### Added

- `--imgsz INT` option for `pedsense train` — input image size for YOLO training (`320`, `640`, `1280`); applies to `yolo`, `yolo-detector`
- `--patience INT` option for `pedsense train` — YOLO early stopping patience (default: `100`); applies to `yolo`, `yolo-detector`
- `--lr FLOAT` option for `pedsense train` — learning rate (default: `1e-4`); applies to `resnet-lstm`, `hybrid`
- `--yolo-epochs INT` option for `pedsense train` — YOLO stage 1 epochs for hybrid pipeline (default: `50`)
- `--device TEXT` option for `pedsense train` — explicit device selection (`'0'`, `'cpu'`, `'0,1'`); applies to all models
- `patience` parameter added to `train_yolo()` and `train_yolo_detector()` API functions

## [1.3.0] - 2026-03-03

### Added

- `pedsense resume` CLI command — interactively list YOLO models and continue training for additional epochs from `weights/last.pt`
- `train -m yolo-detector` — standalone 1-class pedestrian detector training; builds its own dataset internally, no `preprocess yolo` required
- `train_yolo_detector()` function in `pedsense.train.yolo_trainer`
- `train_yolo_resume()` function in `pedsense.train.yolo_trainer`
- `_prepare_detector_data()` shared helper (moved from `hybrid_trainer`); now filters to `PEDESTRIAN_LABELS` only, fixing a bug where non-pedestrian tracks were included in the hybrid detector dataset

### Fixed

- Hybrid pipeline Stage 1 detector dataset incorrectly included all track types (traffic lights, crosswalks) as class 0; now correctly filters to pedestrian-variant tracks only

## [1.2.1] - 2026-03-01

### Added

- `--yolo-variant` option for `pedsense train -m yolo` — select YOLO26 base model size (`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`); default remains `yolo26n`

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
