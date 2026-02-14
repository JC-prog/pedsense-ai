# pedsense setup

Verify and create the project directory structure.

## Synopsis

```bash
uv run pedsense setup
```

## Description

Creates all required directories for the project:

- `data/raw/` — Raw JAAD data (clips, annotations)
- `data/raw/frames/` — Extracted video frames
- `data/processed/yolo/` — YOLO-formatted dataset
- `data/processed/resnet/` — ResNet+LSTM sequences
- `models/base/` — Downloaded pretrained model weights
- `models/custom/` — Trained model outputs

## Example

```bash
$ uv run pedsense setup
Project structure verified!
Place your JAAD raw data in data/raw/
```
