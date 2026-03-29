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

from pedsense.config import (
    DETECTOR_MODELS_DIR,
    DETECTOR_POSE_MODELS_DIR,
    CLASSIFIER_LSTM_MODELS_DIR,
    CLASSIFIER_STGCN_MODELS_DIR,
    CROP_SIZE,
)
from pedsense.train.resnet_lstm import ResNetLSTM, ResNetClassifier, KeypointLSTM

CLASS_NAMES = ["not crossing", "crossing"]
COLORS = {"not crossing": (0, 200, 0), "crossing": (0, 0, 220)}       # BGR: green, red
ACTION_COLORS = {"not crossing": (0, 140, 0), "crossing": (0, 60, 180)}  # darker tones for action label
COLOR_BUFFERING = (0, 180, 255)  # amber — not enough frames yet

# COCO skeleton connections (index pairs into the 17 keypoints)
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


# ---------------------------------------------------------------------------
# Model discovery helpers
# ---------------------------------------------------------------------------

def _find_yolo_weights(model_dir: Path) -> Path | None:
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
    args_file = model_dir / "args.yaml"
    if args_file.exists():
        with open(args_file) as f:
            args = yaml.safe_load(f)
        if "pose" in Path(args.get("model", "")).stem:
            return "yolo-pose"

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type", "yolo")

    if _find_yolo_weights(model_dir) is not None:
        return "yolo"
    if (model_dir / "best.pt").exists():
        return "resnet-lstm"
    return "yolo"


def _list_models() -> dict[str, list[str]]:
    """Scan all model directories and return lists keyed by role.

    Directory layout:
        models/detector/          — YOLO 2-class / 1-class, hybrid
        models/detector-pose/     — YOLO-Pose keypoint detectors
        models/classifier-lstm/   — KeypointLSTM, ResNet-LSTM
        models/classifier-stgcn/  — ST-GCN (future)

    Returns:
        {
            "all_detectors":  detector/ + detector-pose/ entries,
            "pose_detectors": detector-pose/ entries only (for predictor stage 1),
            "intent_models":  classifier-lstm/ + classifier-stgcn/ entries,
        }
    """
    all_detectors: list[str] = []
    pose_detectors: list[str] = []
    intent_models: list[str] = []

    # --- detector/ (YOLO, hybrid) ---
    if DETECTOR_MODELS_DIR.exists():
        for d in sorted(DETECTOR_MODELS_DIR.iterdir(), reverse=True):
            if d.is_dir() and (_find_yolo_weights(d) or (d / "config.json").exists()):
                all_detectors.append(d.name)

    # --- detector-pose/ (YOLO-Pose) ---
    if DETECTOR_POSE_MODELS_DIR.exists():
        for d in sorted(DETECTOR_POSE_MODELS_DIR.iterdir(), reverse=True):
            if d.is_dir() and _find_yolo_weights(d):
                all_detectors.append(d.name)
                pose_detectors.append(d.name)

    # --- classifier-lstm/ ---
    if CLASSIFIER_LSTM_MODELS_DIR.exists():
        for d in sorted(CLASSIFIER_LSTM_MODELS_DIR.iterdir(), reverse=True):
            if d.is_dir() and ((d / "config.json").exists() or (d / "best.pt").exists()):
                intent_models.append(d.name)

    # --- classifier-stgcn/ ---
    if CLASSIFIER_STGCN_MODELS_DIR.exists():
        for d in sorted(CLASSIFIER_STGCN_MODELS_DIR.iterdir(), reverse=True):
            if d.is_dir() and ((d / "config.json").exists() or (d / "best.pt").exists()):
                intent_models.append(d.name)

    return {
        "all_detectors": all_detectors,
        "pose_detectors": pose_detectors,
        "intent_models": intent_models,
    }


def _resolve_model_dir(name: str) -> Path | None:
    for base in (
        DETECTOR_MODELS_DIR,
        DETECTOR_POSE_MODELS_DIR,
        CLASSIFIER_LSTM_MODELS_DIR,
        CLASSIFIER_STGCN_MODELS_DIR,
    ):
        path = base / name
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_yolo_model(model_dir: Path) -> YOLO:
    weights = _find_yolo_weights(model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {model_dir / 'weights'}")
    return YOLO(str(weights))


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
    """Returns (model, input_size, sequence_length)."""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    input_size = cfg.get("input_size", 34)
    sequence_length = cfg.get("sequence_length", 5)
    model = KeypointLSTM(
        input_size=input_size,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(
        torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, input_size, sequence_length


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _box_iou(a: tuple, b: tuple) -> float:
    """IoU for two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _draw_skeleton(frame: np.ndarray, kp: np.ndarray, kp_c: np.ndarray) -> None:
    for a, b in _COCO_SKELETON:
        if kp_c[a] > 0.3 and kp_c[b] > 0.3:
            pa = (int(kp[a][0]), int(kp[a][1]))
            pb = (int(kp[b][0]), int(kp[b][1]))
            if pa != (0, 0) and pb != (0, 0):
                cv2.line(frame, pa, pb, (50, 220, 50), 1)
    for j in range(len(kp)):
        if kp_c[j] > 0.3:
            px, py = int(kp[j][0]), int(kp[j][1])
            if px > 0 or py > 0:
                cv2.circle(frame, (px, py), 3, (0, 200, 255), -1)


def _draw_box_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg_y1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_sub_label(
    frame: np.ndarray,
    x1: int, y2: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a secondary label tag just below the bounding box."""
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    bg_y2 = min(frame.shape[0], y2 + th + 8)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, bg_y2), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, bg_y2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Detector inference
# ---------------------------------------------------------------------------

def _run_detector_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """Run detection-only inference (YOLO 2-class, YOLO-Pose, or Hybrid)."""
    mtype = _detect_model_type(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stats: dict = {"total_detections": 0, "model_type": mtype}

    if mtype == "hybrid":
        yolo, resnet = _load_hybrid_model(model_dir, device)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        yolo = _load_yolo_model(model_dir)
        resnet = None
        transform = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, conf=confidence, verbose=False)

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
                    det_conf = float(box.conf[0])

                    if mtype == "hybrid" and resnet is not None and transform is not None:
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(w, x2), min(h, y2)
                        if x2c > x1c and y2c > y1c:
                            crop = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2RGB)
                            t = transform(crop).unsqueeze(0).to(device)
                            with torch.no_grad():
                                cls_id = resnet(t).argmax(dim=1).item()
                        else:
                            cls_id = 0
                        cls_name = CLASS_NAMES[cls_id]
                        color = COLORS[cls_name]
                        label = f"{cls_name}"

                    elif mtype == "yolo-pose":
                        color = (0, 200, 255)  # cyan for pose-only detector
                        label = f"pedestrian  {det_conf:.0%}"
                        if kpts_data is not None and i < len(kpts_data):
                            kp = kpts_data[i]
                            kp_c = kpts_conf[i] if kpts_conf is not None else np.ones(17)
                            _draw_skeleton(frame, kp, kp_c)

                    else:
                        # YOLO 2-class detector
                        cls_id = int(box.cls[0])
                        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                        color = COLORS.get(cls_name, (200, 200, 200))
                        label = f"{cls_name}  {det_conf:.0%}"

                    _draw_box_label(frame, x1, y1, x2, y2, label, color)
                    stats["total_detections"] += 1

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


# ---------------------------------------------------------------------------
# Predictor inference (Pose + KeypointLSTM)
# ---------------------------------------------------------------------------

def _run_predictor_inference(
    video_path: str,
    pose_model_dir: Path,
    intent_model_dir: Path,
    confidence: float,
    smooth_window: int = 10,
    action_model_dir: Path | None = None,
) -> tuple[str, dict]:
    """2-stage: YOLO-Pose extracts per-track keypoints, KeypointLSTM predicts intent.

    Optionally runs a YOLO action detector on each frame and overlays its
    per-frame crossing/not-crossing classification below each pedestrian box.

    smooth_window controls how many recent LSTM outputs are averaged before the
    crossing/not-crossing label is determined.  Set to 1 to disable smoothing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = _find_yolo_weights(pose_model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {pose_model_dir / 'weights'}")
    pose_model = YOLO(str(weights))

    action_model: YOLO | None = None
    if action_model_dir is not None:
        action_weights = _find_yolo_weights(action_model_dir)
        if action_weights is None:
            raise FileNotFoundError(f"No YOLO weights found in {action_model_dir / 'weights'}")
        action_model = YOLO(str(action_weights))

    lstm, _, sequence_length = _load_keypoint_lstm_model(intent_model_dir, device)

    buffers: dict[int, deque] = {}
    prob_history: dict[int, deque] = {}   # rolling crossing probabilities per track
    last_pred: dict[int, tuple[str, float]] = {}
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
        "crossing_predictions": 0,
        "not_crossing_predictions": 0,
    }

    pose_verified = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pose_model.track(frame, conf=confidence, persist=True, verbose=False)

            # Run action detector on the same frame (independent pass, no tracking needed)
            action_detections: list[tuple[tuple[int,int,int,int], str]] = []
            if action_model is not None:
                for ar in action_model(frame, conf=confidence, verbose=False):
                    for abox in ar.boxes:
                        ax1, ay1, ax2, ay2 = map(int, abox.xyxy[0])
                        cls_id = int(abox.cls[0])
                        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                        action_detections.append(((ax1, ay1, ax2, ay2), cls_name))

            for r in results:
                if r.boxes is None:
                    continue

                if not pose_verified:
                    if r.keypoints is None:
                        raise ValueError(
                            "The selected pose detector does not output keypoints. "
                            "Choose a YOLO-Pose model as the Pose Detector."
                        )
                    pose_verified = True

                track_ids = r.boxes.id
                kpts = r.keypoints

                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    tid = int(track_ids[i]) if track_ids is not None else i

                    # Normalize keypoints using the YOLO-Pose detected bounding box
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
                            prob_history[tid] = deque(maxlen=max(1, smooth_window))
                            stats["pedestrians_tracked"] += 1

                        buffers[tid].append(kp_norm.flatten().astype(np.float32))

                    # Run LSTM once buffer is full
                    if tid in buffers and len(buffers[tid]) == sequence_length:
                        seq = np.stack(list(buffers[tid]))  # (T, 34)
                        tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
                        with torch.no_grad():
                            probs = torch.softmax(lstm(tensor), dim=1)[0]

                        prob_history[tid].append(float(probs[1].item()))
                        smoothed = float(np.mean(list(prob_history[tid])))
                        cls_id = 1 if smoothed > 0.5 else 0
                        cls_name = CLASS_NAMES[cls_id]
                        last_pred[tid] = (cls_name, smoothed)
                        if cls_id == 1:
                            stats["crossing_predictions"] += 1
                        else:
                            stats["not_crossing_predictions"] += 1

                    # Build label and color
                    if tid in last_pred:
                        cls_name, prob = last_pred[tid]
                        color = COLORS[cls_name]
                        label = f"{cls_name}  {prob:.0%}"
                    elif tid in buffers:
                        color = COLOR_BUFFERING
                        label = f"[{len(buffers[tid])}/{sequence_length}]"
                    else:
                        color = COLOR_BUFFERING
                        label = "detecting..."

                    _draw_box_label(frame, x1, y1, x2, y2, label, color)

                    # Action label below box — match by IoU to the action detector output
                    if action_detections:
                        best_iou, best_action_cls = 0.0, None
                        for (ax1, ay1, ax2, ay2), action_cls in action_detections:
                            iou = _box_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                            if iou > best_iou:
                                best_iou, best_action_cls = iou, action_cls
                        if best_action_cls is not None and best_iou > 0.3:
                            action_color = ACTION_COLORS.get(best_action_cls, (100, 100, 100))
                            _draw_sub_label(frame, x1, y2, f"action: {best_action_cls}", action_color)

                    # Skeleton overlay
                    if kpts is not None and i < len(kpts.xy):
                        kp_raw = kpts.xy[i].cpu().numpy()
                        kp_c = (
                            kpts.conf[i].cpu().numpy()
                            if kpts.conf is not None
                            else np.ones(17)
                        )
                        _draw_skeleton(frame, kp_raw, kp_c)

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
    mode: str,
    detector_model_name: str,
    pose_model_name: str,
    intent_model_name: str,
    confidence: float,
    smooth_window: int = 10,
    action_model_name: str | None = None,
) -> tuple[str | None, dict]:
    if not video_path:
        return None, {"error": "Please upload a video."}

    try:
        if mode == "Predictor":
            if not pose_model_name:
                return None, {"error": "Select a Pose Detector for the Predictor pipeline."}
            if not intent_model_name:
                return None, {"error": "Select an Intent Model (KeypointLSTM) for the Predictor pipeline."}
            pose_dir = _resolve_model_dir(pose_model_name)
            intent_dir = _resolve_model_dir(intent_model_name)
            if pose_dir is None:
                return None, {"error": f"Pose model not found: {pose_model_name}"}
            if intent_dir is None:
                return None, {"error": f"Intent model not found: {intent_model_name}"}
            action_dir = _resolve_model_dir(action_model_name) if action_model_name else None
            return _run_predictor_inference(
                video_path, pose_dir, intent_dir, confidence, int(smooth_window), action_dir
            )

        else:  # Detector
            if not detector_model_name:
                return None, {"error": "Select a detector model."}
            det_dir = _resolve_model_dir(detector_model_name)
            if det_dir is None:
                return None, {"error": f"Detector model not found: {detector_model_name}"}
            return _run_detector_inference(video_path, det_dir, confidence)

    except Exception as e:
        return None, {"error": str(e)}


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _refresh_models(mode: str, current_detector: str, current_pose: str, current_intent: str):
    """Re-scan model directories and return updated dropdown components."""
    models = _list_models()
    all_detectors = models["all_detectors"]
    pose_detectors = models["pose_detectors"]
    intent_models = models["intent_models"]
    is_predictor = mode == "Predictor"

    # Keep current selection if it still exists, otherwise pick first available
    def _keep_or_first(current: str, choices: list[str]) -> str | None:
        return current if current in choices else (choices[0] if choices else None)

    return (
        gr.update(
            choices=all_detectors,
            value=_keep_or_first(current_detector, all_detectors),
            visible=not is_predictor,
        ),
        gr.update(
            choices=pose_detectors,
            value=_keep_or_first(current_pose, pose_detectors),
            visible=is_predictor,
        ),
        gr.update(
            choices=intent_models,
            value=_keep_or_first(current_intent, intent_models),
            visible=is_predictor,
        ),
    )


def launch_demo(model_path: str | None = None, port: int = 7860) -> None:
    models = _list_models()
    all_detectors = models["all_detectors"]
    pose_detectors = models["pose_detectors"]
    intent_models = models["intent_models"]

    default_detector = model_path if model_path else (all_detectors[0] if all_detectors else None)
    default_pose = pose_detectors[0] if pose_detectors else None
    default_intent = intent_models[0] if intent_models else None

    def on_mode_change(mode: str, det: str, pose: str, intent: str):
        is_predictor = mode == "Predictor"
        detector_u, pose_u, intent_u = _refresh_models(mode, det, pose, intent)
        predictor_u = gr.update(visible=is_predictor)
        return detector_u, pose_u, intent_u, predictor_u, predictor_u

    with gr.Blocks(title="PedSense-AI Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# PedSense-AI\nPedestrian Crossing Intent Prediction")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")

                mode_radio = gr.Radio(
                    choices=["Detector", "Predictor"],
                    value="Detector",
                    label="Mode",
                )

                # --- Detector mode ---
                detector_dd = gr.Dropdown(
                    choices=all_detectors,
                    value=default_detector,
                    label="Detection Model",
                    visible=True,
                )

                # --- Predictor mode ---
                pose_dd = gr.Dropdown(
                    choices=pose_detectors,
                    value=default_pose,
                    label="Pose Detector",
                    visible=False,
                )
                intent_dd = gr.Dropdown(
                    choices=intent_models,
                    value=default_intent,
                    label="Intent Model (KeypointLSTM)",
                    visible=False,
                )

                action_dd = gr.Dropdown(
                    choices=all_detectors,
                    value=None,
                    label="Action Detector — optional (YOLO 2-class)",
                    info="If set, overlays a per-frame crossing/not-crossing label below each bounding box.",
                    visible=False,
                )

                confidence_slider = gr.Slider(
                    0.1, 1.0, value=0.5, step=0.05,
                    label="Confidence Threshold",
                )

                smooth_slider = gr.Slider(
                    1, 30, value=10, step=1,
                    label="Prediction Smoothing (frames)",
                    info="Average crossing probability over last N outputs. 1 = no smoothing, 10 = ~1s average.",
                    visible=False,
                )

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    refresh_btn = gr.Button("Refresh Models", variant="secondary")

            with gr.Column(scale=2):
                video_output = gr.Video(label="Output")
                stats_output = gr.JSON(label="Statistics")

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio, detector_dd, pose_dd, intent_dd],
            outputs=[detector_dd, pose_dd, intent_dd, action_dd, smooth_slider],
        )

        refresh_btn.click(
            fn=_refresh_models,
            inputs=[mode_radio, detector_dd, pose_dd, intent_dd],
            outputs=[detector_dd, pose_dd, intent_dd],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                video_input, mode_radio, detector_dd,
                pose_dd, intent_dd,
                confidence_slider, smooth_slider,
                action_dd,
            ],
            outputs=[video_output, stats_output],
        )

    app.launch(server_port=port, share=False)
