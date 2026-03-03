# pedsense resume

Continue training an existing YOLO model for additional epochs.

## Synopsis

```bash
uv run pedsense resume
```

## Description

`resume` is an interactive command that:

1. Lists all YOLO models in `models/custom/` that have a `weights/last.pt` checkpoint
2. Prompts you to select a model by number
3. Prompts for how many additional epochs to train
4. Resumes training from `last.pt` and saves the result to a new directory

Only YOLO-based models (`yolo`, `yolo-detector`) are listed. ResNet+LSTM and Hybrid models use a different checkpoint format and are not shown.

## Interactive Flow

```
$ uv run pedsense resume

        Resumable YOLO Models
 # │ Name                          │ Type          │ Variant │ Epochs │ mAP50
───┼───────────────────────────────┼───────────────┼─────────┼────────┼──────
 1 │ my_detector_20260301_224648   │ yolo-detector │ yolo26m │     10 │ 0.803
 2 │ my_detector_20260301_224937   │ yolo-detector │ yolo26m │     10 │ 0.810
 3 │ my_yolo_20260301_202546       │ yolo          │ yolo26m │     30 │ 0.743

Select model number: 2
Selected: my_detector_20260301_224937 (10 epochs trained so far)
Additional epochs to train: 10
Training 10 more epochs → total 20 epochs
...
Resumed model saved to: models/custom/my_detector_20260301_224937_resumed_20260302_100000
```

## Output

The resumed model is saved to a new directory:

```
models/custom/{original_name}_resumed_{YYYYMMDD_HHMMSS}/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
└── args.yaml
```

The original model directory is left unchanged.

## How Resuming Works

Training is continued from `weights/last.pt` of the selected model. The data path and hyperparameters (`batch`, `imgsz`) are read automatically from the original `args.yaml` — no manual configuration required.

!!! note
    This adds **new** epochs on top of the completed run, not just resumes an interrupted one. Each resume creates a fresh output directory so you can compare runs.

## Typical Workflow

```bash
# Initial training (10 epochs)
uv run pedsense train -m yolo-detector -n my_detector -e 10 --yolo-variant yolo26m

# Evaluate results, then add more epochs
uv run pedsense resume
# → select my_detector_*, enter 40 → trains epochs 11–50
# → new dir: my_detector_*_resumed_*/

# Chain resumes if needed
uv run pedsense resume
# → select the resumed model, add more epochs
```

## Chaining with Hybrid

Once you have a well-trained detector, pass it to the hybrid pipeline to skip re-training:

```bash
uv run pedsense train -m hybrid -n my_hybrid \
  --yolo-model models/custom/my_detector_20260301_224937_resumed_20260302_100000/weights/best.pt
```
