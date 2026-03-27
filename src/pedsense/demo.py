import json
import tempfile
from collections import deque
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from torchvision import transforms
from ultralytics import YOLO

from pedsense.config import DETECTOR_MODELS_DIR, CLASSIFIER_MODELS_DIR, CROP_SIZE
from pedsense.train.resnet_lstm import ResNetLSTM, ResNetClassifier, KeypointLSTM

CLASS_NAMES = ["not-crossing", "crossing"]
COLORS = {"not-crossing": (0, 255, 0), "crossing": (0, 0, 255)}  # BGR: green, red
COLOR_BUFFERING = (0, 215, 255)  # yellow — not enough frames yet

# COCO skeleton connections (index pairs into the 17 keypoints)
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


def _find_yolo_weights(model_dir: Path) -> Path | None:
    """Find the best available YOLO weights file in an Ultralytics output directory."""
    weights_dir = model_dir / "weights"
    if not weights_dir.is_dir():
        return None
    for name in ("best.pt", "last.pt"):
        candidate = weights_dir / name
        if candidate.exists():
            return candidate
    pt_files = list(weights_dir.glob("*.pt"))
    return pt_files[0] if pt_files else None


def _detect_model_type(model_dir: Path) -> str:
    """Detect model type from config.json, args.yaml, or directory contents."""
    args_file = model_dir / "args.yaml"
    if args_file.exists():
        with open(args_file) as f:
            args = yaml.safe_load(f)
        model_path = args.get("model", "")
        if "pose" in Path(model_path).stem:
            return "yolo-pose"

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type", "yolo")

    if _find_yolo_weights(model_dir) is not None:
        return "yolo"
    if (model_dir / "best.pt").exists():
        return "resnet-lstm"
    return "yolo"


def _list_models_by_type() -> dict[str, list[str]]:
    """Return model names grouped by type: detection, keypoint-lstm, all."""
    result: dict[str, list[str]] = {"detection": [], "keypoint-lstm": [], "all": []}
    for base_dir in (DETECTOR_MODELS_DIR, CLASSIFIER_MODELS_DIR):
        if not base_dir.exists():
            continue
        for d in sorted(base_dir.iterdir(), reverse=True):
            if not (d.is_dir() and ((d / "config.json").exists() or _find_yolo_weights(d))):
                continue
            mtype = _detect_model_type(d)
            result["all"].append(d.name)
            if mtype in ("yolo", "yolo-pose", "hybrid"):
                result["detection"].append(d.name)
            elif mtype == "keypoint-lstm":
                result["keypoint-lstm"].append(d.name)
    return result


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_yolo_model(model_dir: Path) -> YOLO:
    weights = _find_yolo_weights(model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {model_dir / 'weights'}")
    return YOLO(str(weights))


def _resolve_model_dir(name: str) -> Path | None:
    """Find which base directory a model lives in."""
    for base in (DETECTOR_MODELS_DIR, CLASSIFIER_MODELS_DIR):
        path = base / name
        if path.exists():
            return path
    return None


def _load_resnet_lstm_model(model_dir: Path, device: torch.device) -> ResNetLSTM:
    model = ResNetLSTM(num_classes=2)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _load_hybrid_model(model_dir: Path, device: torch.device) -> tuple[YOLO, ResNetClassifier]:
    yolo = YOLO(str(model_dir / "yolo_detector.pt"))
    resnet = ResNetClassifier(num_classes=2)
    resnet.load_state_dict(
        torch.load(model_dir / "resnet_classifier.pt", map_location=device, weights_only=True)
    )
    resnet.to(device)
    resnet.eval()
    return yolo, resnet


def _load_keypoint_lstm_model(
    model_dir: Path, device: torch.device
) -> tuple[KeypointLSTM, int, int]:
    """Load KeypointLSTM. Returns (model, input_size, sequence_length)."""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    input_size = cfg.get("input_size", 34)
    sequence_length = cfg.get("sequence_length", 16)
    model = KeypointLSTM(
        input_size=input_size,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, input_size, sequence_length


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def _run_yolo_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """Run YOLO detection-only inference."""
    model = _load_yolo_model(model_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stats = {"total_detections": 0, "crossing": 0, "not_crossing": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                    color = COLORS.get(cls_name, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    stats["total_detections"] += 1
                    if cls_name == "crossing":
                        stats["crossing"] += 1
                    else:
                        stats["not_crossing"] += 1

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


def _run_hybrid_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """Hybrid: YOLO detects bounding boxes, ResNet classifies each crop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo, resnet = _load_hybrid_model(model_dir, device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stats = {"total_detections": 0, "crossing": 0, "not_crossing": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, conf=confidence, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = resnet(input_tensor)
                        cls_id = logits.argmax(dim=1).item()

                    cls_name = CLASS_NAMES[cls_id]
                    color = COLORS.get(cls_name, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    stats["total_detections"] += 1
                    if cls_name == "crossing":
                        stats["crossing"] += 1
                    else:
                        stats["not_crossing"] += 1

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


def _run_yolo_pose_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """YOLO-Pose: render skeleton only, no intent classification."""
    weights = _find_yolo_weights(model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {model_dir / 'weights'}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stats = {"total_pedestrians": 0, "model_type": "yolo-pose"}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                kpts_data = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
                kpts_conf = (
                    r.keypoints.conf.cpu().numpy()
                    if (r.keypoints is not None and r.keypoints.conf is not None)
                    else None
                )

                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "pedestrian", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    stats["total_pedestrians"] += 1

                    if kpts_data is None or i >= len(kpts_data):
                        continue

                    kp = kpts_data[i]
                    kp_c = kpts_conf[i] if kpts_conf is not None else np.ones(17)

                    for a, b in _COCO_SKELETON:
                        if kp_c[a] > 0.3 and kp_c[b] > 0.3:
                            pt_a = (int(kp[a][0]), int(kp[a][1]))
                            pt_b = (int(kp[b][0]), int(kp[b][1]))
                            if pt_a != (0, 0) and pt_b != (0, 0):
                                cv2.line(frame, pt_a, pt_b, (0, 255, 0), 2)

                    for j in range(17):
                        if kp_c[j] > 0.3:
                            px, py = int(kp[j][0]), int(kp[j][1])
                            if px > 0 or py > 0:
                                cv2.circle(frame, (px, py), 4, (0, 215, 255), -1)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


def _run_keypoint_lstm_inference(
    video_path: str,
    pose_model_dir: Path,
    lstm_model_dir: Path,
    confidence: float,
) -> tuple[str, dict]:
    """2-stage: YOLO-Pose extracts keypoints per tracked pedestrian,
    KeypointLSTM classifies crossing intent once T frames are buffered.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = _find_yolo_weights(pose_model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {pose_model_dir / 'weights'}")
    pose_model = YOLO(str(weights))

    lstm, _, sequence_length = _load_keypoint_lstm_model(lstm_model_dir, device)

    # Per-track state: deque of flattened (17*2,) normalized keypoint vectors
    buffers: dict[int, deque] = {}
    last_pred: dict[int, tuple[str, float]] = {}  # tid -> (class_name, prob)
    seen_ids: set[int] = set()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stats = {
        "model_type": "keypoint-lstm",
        "sequence_length": sequence_length,
        "pedestrians_tracked": 0,
        "crossing": 0,
        "not_crossing": 0,
    }

    pose_model_verified = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Use .track() so each pedestrian keeps a consistent ID across frames
            results = pose_model.track(frame, conf=confidence, persist=True, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                # Fail fast if the selected model has no keypoint output
                if not pose_model_verified:
                    if r.keypoints is None:
                        raise ValueError(
                            "The selected pose detector does not output keypoints. "
                            "Select a YOLO-Pose model (e.g. trained with 'train -m yolo-pose') "
                            "as the Pose Detector."
                        )
                    pose_model_verified = True

                track_ids = r.boxes.id
                kpts = r.keypoints

                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    tid = int(track_ids[i]) if track_ids is not None else i

                    # Normalize keypoints relative to bounding box
                    if kpts is not None and i < len(kpts.xy):
                        kp = kpts.xy[i].cpu().numpy()  # (17, 2)
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        bh = max(y2 - y1, 1)
                        kp_norm = kp.copy()
                        kp_norm[:, 0] = (kp[:, 0] - cx) / bh
                        kp_norm[:, 1] = (kp[:, 1] - cy) / bh

                        if tid not in seen_ids:
                            seen_ids.add(tid)
                            buffers[tid] = deque(maxlen=sequence_length)
                            stats["pedestrians_tracked"] += 1

                        buffers[tid].append(kp_norm.flatten().astype(np.float32))

                    # Classify when the buffer has enough frames
                    if tid in buffers and len(buffers[tid]) == sequence_length:
                        seq = np.stack(list(buffers[tid]))  # (T, 34)
                        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
                        with torch.no_grad():
                            probs = torch.softmax(lstm(tensor), dim=1)[0]
                            cls_id = int(probs.argmax().item())
                        cls_name = CLASS_NAMES[cls_id]
                        last_pred[tid] = (cls_name, float(probs[cls_id].item()))
                        if cls_id == 1:
                            stats["crossing"] += 1
                        else:
                            stats["not_crossing"] += 1

                    # Draw bounding box + label
                    if tid in last_pred:
                        cls_name, prob = last_pred[tid]
                        color = COLORS[cls_name]
                        label = f"ID:{tid} {cls_name} {prob:.2f}"
                    elif tid in buffers:
                        color = COLOR_BUFFERING
                        label = f"ID:{tid} buffering {len(buffers[tid])}/{sequence_length}"
                    else:
                        color = COLOR_BUFFERING
                        label = f"ID:{tid} no keypoints"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Draw skeleton
                    if kpts is not None and i < len(kpts.xy):
                        kp_raw = kpts.xy[i].cpu().numpy()
                        kp_c = (
                            kpts.conf[i].cpu().numpy()
                            if kpts.conf is not None
                            else np.ones(17)
                        )
                        for a, b in _COCO_SKELETON:
                            if kp_c[a] > 0.3 and kp_c[b] > 0.3:
                                pa = (int(kp_raw[a][0]), int(kp_raw[a][1]))
                                pb = (int(kp_raw[b][0]), int(kp_raw[b][1]))
                                if pa != (0, 0) and pb != (0, 0):
                                    cv2.line(frame, pa, pb, (0, 255, 0), 1)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def run_inference(
    video_path: str,
    pipeline: str,
    detection_model_name: str,
    intent_model_name: str | None,
    confidence: float,
) -> tuple[str | None, dict]:
    """Route inference to the appropriate runner based on pipeline selection."""
    if not video_path or not detection_model_name:
        return None, {"error": "Please select a video and a detection model."}

    detection_dir = _resolve_model_dir(detection_model_name)
    if detection_dir is None:
        return None, {"error": f"Model not found: {detection_model_name}"}

    if pipeline == "2-Stage Intent (Pose + LSTM)":
        if not intent_model_name:
            return None, {"error": "Select a KeypointLSTM intent model for 2-stage inference."}
        intent_dir = _resolve_model_dir(intent_model_name)
        if intent_dir is None:
            return None, {"error": f"Intent model not found: {intent_model_name}"}
        try:
            return _run_keypoint_lstm_inference(video_path, detection_dir, intent_dir, confidence)
        except Exception as e:
            return None, {"error": str(e)}

    # Detection-only
    model_type = _detect_model_type(detection_dir)
    try:
        if model_type == "yolo":
            return _run_yolo_inference(video_path, detection_dir, confidence)
        elif model_type == "yolo-pose":
            return _run_yolo_pose_inference(video_path, detection_dir, confidence)
        elif model_type == "hybrid":
            return _run_hybrid_inference(video_path, detection_dir, confidence)
        elif model_type == "keypoint-lstm":
            return None, {
                "error": "KeypointLSTM requires a pose detector. "
                "Switch to '2-Stage Intent (Pose + LSTM)' pipeline and select a pose model."
            }
        elif model_type == "resnet-lstm":
            return None, {
                "error": "ResNet+LSTM requires a separate pedestrian detector. "
                "Use a YOLO or Hybrid model for detection-only, or train a Hybrid model."
            }
        else:
            return None, {"error": f"Unknown model type: {model_type}"}
    except Exception as e:
        return None, {"error": str(e)}


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def launch_demo(model_path: str | None = None, port: int = 7860) -> None:
    """Launch Gradio interface for pedestrian intent prediction."""
    models_by_type = _list_models_by_type()
    detection_models = models_by_type["detection"]
    lstm_models = models_by_type["keypoint-lstm"]

    default_detection = model_path if model_path else (detection_models[0] if detection_models else None)
    default_lstm = lstm_models[0] if lstm_models else None

    def on_pipeline_change(pipeline: str):
        is_2stage = pipeline == "2-Stage Intent (Pose + LSTM)"
        return (
            gr.update(
                label="Pose Detector" if is_2stage else "Model",
                choices=detection_models,
                value=detection_models[0] if detection_models else None,
            ),
            gr.update(visible=is_2stage),
        )

    with gr.Blocks(title="PedSense-AI Demo") as app:
        gr.Markdown("# PedSense-AI: Pedestrian Crossing Intent Prediction")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")

                pipeline_selector = gr.Radio(
                    choices=["Detection Only", "2-Stage Intent (Pose + LSTM)"],
                    value="Detection Only",
                    label="Pipeline",
                )

                detection_model_dd = gr.Dropdown(
                    choices=detection_models,
                    value=default_detection,
                    label="Model",
                )

                intent_model_dd = gr.Dropdown(
                    choices=lstm_models,
                    value=default_lstm,
                    label="Intent Model (KeypointLSTM)",
                    visible=False,
                )

                confidence_slider = gr.Slider(
                    0.1, 1.0, value=0.5, step=0.05,
                    label="Confidence Threshold",
                )
                run_button = gr.Button("Run Prediction", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Annotated Output")
                stats_output = gr.JSON(label="Statistics")

        pipeline_selector.change(
            fn=on_pipeline_change,
            inputs=pipeline_selector,
            outputs=[detection_model_dd, intent_model_dd],
        )

        run_button.click(
            fn=run_inference,
            inputs=[video_input, pipeline_selector, detection_model_dd, intent_model_dd, confidence_slider],
            outputs=[video_output, stats_output],
        )

    app.launch(server_port=port, share=False)
