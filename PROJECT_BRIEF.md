# Pedestrian Crossing Intention Prediction — Project Brief

## Project Goal

Predict whether a pedestrian intends to cross the road using a two-stage system:
1. **Stage 1 — Detection**: YOLOv8-Pose detects pedestrians and extracts 17-joint COCO skeleton keypoints `(x, y)` per frame
2. **Stage 2 — Prediction**: A sequence model classifies crossing intention from a temporal window of keypoint sequences

---

## Dataset: JAAD (Joint Attention in Autonomous Driving)

- **346 short video clips** (5–15s) from dashboard-mounted cameras, recorded in North America and Europe
- **Two usable subsets**:
  - `JAADbeh` — ~650 pedestrians with full behavior annotations (cleaner, smaller)
  - `JAADall` — ~2,100 samples including bystanders (larger, more imbalanced)
- **Key annotation fields per pedestrian per frame**:
  - `bbox`: `[x1, y1, x2, y2]`
  - `occlusion`: `0` (none), `1` (partial >25%), `2` (full >75%)
  - `behavior.cross`: `1` (crosses), `0` (does not cross), `-1` (no intention)
  - `behavior.action`, `behavior.look`, `behavior.nod`, `behavior.hand_gesture`
  - `crossing_point`: frame index when pedestrian starts crossing (or last visible frame if no crossing)
- **Important caveat**: JAAD intention labels are based on observed actions, not validated human-annotated intent. Labels reflect "did they cross?" not "were they about to cross?" Use `crossing_point` as the temporal anchor for prediction windows.
- **Pose keypoints are NOT in the original JAAD annotations** — they must be extracted offline using YOLO-Pose and aligned to JAAD pedestrian track IDs.

---

## Current Pipeline

```
JAAD video clips
    └── Extract frames (split_clips_to_frames.sh)
              │
              ▼
    Run YOLO-Pose offline on all training frames
              │
              ▼
    Align YOLO-Pose detections → JAAD pedestrian IDs (by bbox IoU or track)
              │
              ▼
    Build sliding window sequences relative to crossing_point
    (e.g. frames [t-N ... t-1] → binary label: crossing / not crossing)
              │
              ▼
    Train Stage 2 model (currently LSTM/GRU on flattened keypoints)
              │
              ▼
    Inference: live YOLO-Pose → Stage 2 → crossing prediction
```

---

## Stage 2 Model — Current vs Candidate

### Current: LSTM/GRU
- Input: flattened keypoint vector per frame → `(T, 34)` for 17 joints × 2 coords
- Problem: loses spatial structure between joints (treats left knee and right shoulder as equally related)

### Candidate: ST-GCN (Spatial-Temporal Graph Convolutional Network)
- Paper: [arXiv:1801.07455](https://arxiv.org/abs/1801.07455)
- Models skeleton as a graph: nodes = joints, spatial edges = anatomical connections, temporal edges = same joint across frames
- Input shape: `(batch, channels, T, V)` — channels=2 (x,y), T=frames, V=17 joints
- Adjacency matrix: hardcoded from COCO 17-joint skeleton connections (no need to learn)
- Advantages: explicit structural inductive bias, better data efficiency, more expressive temporal+spatial modeling

---

## Key Design Decisions to Investigate

### 1. Keypoint Alignment
- How are YOLO-Pose detections matched to JAAD pedestrian IDs?
- Is IoU matching used? Is there a tracking step (e.g. ByteTrack, DeepSORT)?
- What happens during occlusion frames (`occlusion > 0`)?

### 2. Sequence Construction
- What is the observation window length `T`? (typical: 10–30 frames)
- How are sequences sampled relative to `crossing_point`?
- Is there overlap between sequences? What stride is used?
- How are samples from JAADbeh vs JAADall handled?

### 3. Normalization
- Are keypoints normalized relative to bounding box (center + scale)?
- Raw pixel coordinates are view-dependent — a pedestrian far away looks different from one close up
- Recommended: normalize each joint relative to bbox center and bbox height

### 4. Class Imbalance
- JAADall skews heavily toward non-crossing samples
- Is class weighting, oversampling, or stratified splitting applied?
- Is the train/test split pedestrian-level (not frame-level) to prevent leakage?

### 5. Occlusion Handling
- Frames with heavy occlusion (`occlusion == 2`) may have unreliable keypoints
- Are occluded frames masked, dropped, or imputed?

---

## Known Limitations of JAAD

- No ego-vehicle speed/motion data (unlike PIE dataset)
- Videos are short discontinuous clips — no long continuous tracks
- Intention labels are action-derived, not human-validated
- Pose annotations must be generated externally (YOLO-Pose, OpenPose, ViTPose)
- Geographic/environmental bias: North America + Europe only

---

## Questions for Claude Code to Answer

When analyzing the codebase, focus on:

1. **Data pipeline**: How is `jaad_data.py` (or equivalent) loading and structuring annotations? Is `crossing_point` used as the label anchor?
2. **Keypoint extraction**: Where and how is YOLO-Pose run on JAAD frames? What is the alignment strategy?
3. **Sequence builder**: How are input sequences `(T, V, C)` constructed? What is T, what normalization is applied?
4. **Model architecture**: What is the exact LSTM/GRU input/output shape? Where is the classifier head?
5. **Training setup**: Loss function, class weighting, optimizer, train/val/test split strategy
6. **Evaluation metrics**: Accuracy, F1, AUC — are they computed on balanced or raw distributions?
7. **ST-GCN readiness**: What changes are needed to swap the LSTM for ST-GCN? Is the data loader already in `(T, V, C)` format or does it flatten joints?

---

## Suggested Next Steps (Priority Order)

1. Audit the keypoint-to-track alignment logic for correctness
2. Verify sequences are split at the **pedestrian level** (not frame level) to avoid data leakage
3. Add bounding-box-relative keypoint normalization if not already present
4. Evaluate current LSTM baseline on JAADbeh with standard metrics (Acc, F1, AUC)
5. Implement ST-GCN as a drop-in replacement for the LSTM stage
6. Compare both models on JAADbeh and JAADall

---

## Reference Papers

- Rasouli et al., "Are They Going to Cross?" ICCVW 2017 — original JAAD paper
- Yan et al., "ST-GCN", AAAI 2018 — [arXiv:1801.07455](https://arxiv.org/abs/1801.07455)
- Kotseruba et al., "Benchmark for Evaluating Pedestrian Action Prediction", WACV 2021
- Rasouli et al., "PIE", ICCV 2019 — useful contrast dataset with stronger intention labels
- Ghiya et al., "SGNetPose+", 2025 — recent SOTA using pose + bounding box on JAAD
