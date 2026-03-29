# Using Pretrained Models

If you have downloaded a PedSense model from Hugging Face (or any other source), this page shows you exactly where to place the files so the demo can discover and load them.

## How the Demo Finds Models

The demo scans two directories at startup:

| Directory | Model types stored here |
|-----------|------------------------|
| `models/detector/` | YOLO26, YOLO-Pose, Hybrid |
| `models/classifier/` | KeypointLSTM, ResNet+LSTM |

Each model must live in its own **named subfolder** inside one of these directories. The subfolder name becomes the model's display name in the demo dropdown.

A model is listed only if it contains:

- A `weights/` subfolder with at least one `.pt` file **(YOLO / YOLO-Pose)**, or
- A `config.json` file **(Hybrid, KeypointLSTM, ResNet+LSTM)**

---

## File Layout by Model Type

### YOLO26 / YOLO-Pose

```
models/
└── detector/
    └── my_yolo_model/          ← your chosen folder name
        └── weights/
            └── best.pt         ← the downloaded .pt file
```

The demo looks for `best.pt` first, then `last.pt`, then any other `.pt` in the `weights/` folder.

### Hybrid (YOLO detector + ResNet classifier)

```
models/
└── detector/
    └── my_hybrid_model/
        ├── yolo_detector.pt
        ├── resnet_classifier.pt
        └── config.json         ← must include "model_type": "hybrid"
```

### KeypointLSTM

```
models/
└── classifier/
    └── my_keypoint_lstm/
        ├── best.pt
        └── config.json         ← must include "model_type": "keypoint-lstm"
                                   and "sequence_length": <T>
```

### ResNet+LSTM

```
models/
└── classifier/
    └── my_resnet_lstm/
        ├── best.pt
        └── config.json         ← must include "model_type": "resnet-lstm"
```

---

## Downloading with the CLI

The easiest way to install a model is with `pedsense download`. It fetches the full Hugging Face repo, detects the model type, and places the files in the correct directory automatically.

```bash
# Download by repo ID — local folder name defaults to the repo name
uv run pedsense download JCProg/pedsense-yolo

# Use a custom local name
uv run pedsense download JCProg/pedsense-yolo --name my_yolo

# Overwrite an existing folder
uv run pedsense download JCProg/pedsense-yolo --force
```

Then launch the demo — the model will appear in the dropdown immediately:

```bash
uv run pedsense demo
# or pre-select it
uv run pedsense demo -m my_yolo
```

---

## Manual Installation (if needed)

If you have model files locally (e.g. downloaded outside of `pedsense download`), place them according to the layouts above, then run:

```bash
uv run pedsense demo
```

---

## Verifying Discovery

You can check which models the demo has found before launching the full UI:

```python
# from the project root
uv run python -c "
from pedsense.demo import _list_models_by_type
models = _list_models_by_type()
print('Detection models:', models['detection'])
print('KeypointLSTM models:', models['keypoint-lstm'])
"
```

If a model is missing from the output, check:

- The folder is inside `models/detector/` or `models/classifier/` (not directly in `models/`)
- YOLO models have a `weights/` subfolder containing at least one `.pt` file
- Hybrid / KeypointLSTM / ResNet+LSTM models have a `config.json` with the correct `model_type` field

---

## Complete Reference

| Model type | Parent directory | Required files |
|------------|-----------------|----------------|
| YOLO26 | `models/detector/` | `weights/best.pt` (or `last.pt`) |
| YOLO-Pose | `models/detector/` | `weights/best.pt` (or `last.pt`) |
| Hybrid | `models/detector/` | `yolo_detector.pt`, `resnet_classifier.pt`, `config.json` |
| KeypointLSTM | `models/classifier/` | `best.pt`, `config.json` |
| ResNet+LSTM | `models/classifier/` | `best.pt`, `config.json` |
