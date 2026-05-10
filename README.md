# 🎾 Padel Shot Classification using Computer Vision

A real-time padel shot classification system that combines **YOLOv8 pose estimation**, **YOLOv8 object detection**, and a custom-trained **Bidirectional GRU model** to classify player shots — backhand, forehand, ready position, and serve — directly from video input.

---

Note: This project is a prototype and does not guarantee accurate results in all scenarios — future improvements such as more precise GRU training on padel-specific data and better ball/racket detection models can significantly enhance performance.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
  - [Pose Estimation — YOLOv8s-Pose](#pose-estimation--yolov8s-pose)
  - [Object Detection — YOLOv8x](#object-detection--yolov8x)
  - [Shot Classifier — Bidirectional GRU](#shot-classifier--bidirectional-gru)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
  - [Local Inference](#local-inference-srcinferencepy)
  - [Google Colab Inference](#google-colab-inference-inference_colabipynb)
- [Installation](#installation)
- [Challenges Faced](#challenges-faced)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project performs **real-time padel shot classification** from video footage using a multi-model pipeline:

1. **YOLOv8s-Pose** extracts 17 body keypoints per player per frame with ByteTrack player tracking.
2. **YOLOv8x** detects the ball and racket in each frame.
3. A custom-trained **Bidirectional GRU** classifies the shot type from a rolling window of 20 frames of keypoint sequences.
4. A **HUD overlay** displays shot counts, player labels, and confidence scores in real time.

The system supports multi-player tracking and produces a JSON log of all detected shot events. A Google Colab notebook is also provided that runs the full pipeline headlessly and saves the annotated output as a video file.

---

## Demo

The inference pipeline overlays the following on each video frame:

- **Skeleton** drawn on each tracked player (color-coded per player ID)
- **Shot label + confidence** shown above each player bounding box (e.g. `P1 | FOREHAND  87.3%`)
- **Ball** highlighted as a filled cyan circle
- **Racket** highlighted with a green bounding box
- **Shot count HUD** in the bottom-right corner (backhand / forehand / serve totals)

---

## Architecture

```
Video Frame
    │
    ├──► YOLOv8s-Pose ──► 17 Keypoints per Player (ByteTrack ID)
    │                           │
    │                    + Synthetic Neck Keypoint (avg. of shoulders)
    │                           │
    │                    Hip-center normalize → scale → flatten → 54-dim vector
    │                           │
    │                    Rolling Buffer (last 20 frames per player)
    │                           │
    │                    Bidirectional GRU ──► Softmax ──► Shot Label + Confidence
    │
    └──► YOLOv8x ──► Ball & Racket Detection ──► Overlay on Frame
```

---

## Project Structure

```
padel-shot-classification/
│
├── src/
│   ├── model.py                        # PadelGRU model definition
│   ├── prepare_dataset.py              # Dataset class + augmentation pipeline
│   ├── train.py                        # GRU training loop + checkpointing
│   ├── evaluation.py                   # Validation metrics + confusion matrix
│   └── inference.py                    # Real-time local inference (cv2.imshow)
│
├── inference_colab.ipynb               # Colab notebook — headless, saves output.mp4
│
├── data/
│   ├── input_sample_video.mp4          # Input video for inference
│   └── annotations/
│       ├── backhand.json               # COCO-format keypoint annotations
│       ├── forehand.json
│       ├── ready_position.json
│       └── serve.json
│
├── models/
│   └──custom_gru.pth            # Saved GRU model weights
│
├── outputs/
│   ├── shot_results.json               # Per-frame shot event log
│   ├── classification_report.json      # Per-class precision / recall / F1
│   └── confusion_matrix.png            # Confusion matrix heatmap
│
└── requirements.txt
```

---

## Dataset

The GRU classifier was trained on the **Tennis Player Actions Dataset** from Kaggle:

> 📦 [https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset](https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset)

The dataset provides keypoint annotations (COCO format, 17 joints) across four action classes:

| Class Index | Class Name      |
|-------------|-----------------|
| 0           | Backhand        |
| 1           | Forehand        |
| 2           | Ready Position  |
| 3           | Serve           |

Annotations are stored as JSON files under `data/annotations/<class_name>.json`. Each annotation contains an `image_id` and a flat `keypoints` array representing 18 joints × 3 values (x, y, visibility) — the 18th being a synthetic neck joint added during preprocessing.

**Train / Validation split:** 80% / 20%, split chronologically (not shuffled) to preserve temporal structure.

---

## Models

### Pose Estimation — YOLOv8s-Pose

- **Model:** `yolov8s-pose.pt` (pretrained, Ultralytics) — auto-downloaded on first run
- **Purpose:** Detect and track players, extract 17 COCO body keypoints per frame
- **Tracking:** ByteTrack via `.track(persist=True)` for consistent player IDs across frames
- **Output:** 17 × 3 keypoints (x, y, confidence) per tracked player

### Object Detection — YOLOv8x

- **Model:** `yolov8x.pt` (pretrained COCO, Ultralytics) — auto-downloaded on first run
- **Purpose:** Detect tennis ball (`sports ball`) and racket (`tennis racket`) in each frame
- **Classes used:** Only `sports ball` and `tennis racket` are rendered; all other COCO classes are filtered out

> See [Challenges Faced](#challenges-faced) for the full story on why the pretrained model was ultimately used over a custom-trained one.

### Shot Classifier — Bidirectional GRU

**Model definition:** `src/model.py`

```
Input  →  [batch, seq_len=20, input_size=54]
           ↓
           Bidirectional GRU
           (hidden=16, layers=2, dropout=0.3, bidirectional=True)
           ↓
           Mean Pooling over time dimension  →  [batch, 32]
           ↓
           Linear(32 → 64) + ReLU + Dropout(0.4)
           ↓
           Linear(64 → 4)
Output →  [batch, num_classes=4]
```

**Key design choices:**

- **Bidirectional** — captures both past and future context within a motion sequence, useful for actions that have distinct wind-up and follow-through phases
- **Mean pooling** — more stable than using only the last hidden state, especially across variable-length motion patterns
- **Input size = 54** — 18 keypoints × 3 values (x, y, visibility). The 18th keypoint is a synthetic neck computed at inference time as the midpoint of the left and right shoulder keypoints (COCO indices 5 and 6)
- **Sequence length = 20 frames** — rolling buffer updated every frame, GRU classification triggered every 3 frames

---

## Data Preprocessing & Augmentation

**Preprocessing** is applied identically during both training (`src/prepare_dataset.py`) and inference (`src/inference.py` / `inference_colab.ipynb`):

1. **Missing keypoint zeroing** — keypoints with visibility = 0 have their (x, y) set to (0.0, 0.0)
2. **Hip-center normalization** — all keypoints are translated so the midpoint between the left and right hip is the origin, making the representation invariant to body position in the frame
3. **Resolution scaling** — x divided by 1280, y divided by 720 (assumed video resolution)
4. **Flattening** — 18 × 3 array → 54-dimensional vector per frame

**Augmentation** is applied to training sequences only:

| Augmentation   | Probability | Description                                      |
|----------------|-------------|--------------------------------------------------|
| Gaussian Noise | 50%         | `std=0.01` additive noise on keypoint values    |
| Random Scale   | 30%         | Uniform scale factor sampled from `[0.9, 1.1]` |
| Temporal Shift | 30%         | Rolls the sequence by up to ±2 frames            |
| Joint Dropout  | 20%         | Randomly zeros out ~10% of keypoint values       |
| Time Jitter    | 20%         | Randomly permutes the frame order in a sequence  |

---

## Training

**Hyperparameters:**

| Parameter      | Value            |
|----------------|------------------|
| `input_size`   | 54               |
| `hidden_size`  | 16               |
| `num_layers`   | 2                |
| `num_classes`  | 4                |
| `seq_len`      | 20               |
| `batch_size`   | 16               |
| `epochs`       | 20               |
| `lr`           | 3e-4             |
| `weight_decay` | 1e-4             |
| Optimizer      | AdamW            |
| Loss           | CrossEntropyLoss |

**Sequence generation:** Sliding window over sorted annotation frames with stride = 2.

**Model checkpointing:** The model with the best validation accuracy is saved to `models/custom_gru.pth`.

```bash
cd src
python train.py
```

Sample training output:

```
Epoch 1/20 Training Loss: X.XXXX Training accuracy: XX.XX Validation Loss: X.XXXX Validation accuracy: XX.XX
Best model saved (XX.XX%)
```

---

## Evaluation

Run evaluation on the validation split to generate a per-class report and confusion matrix:

```bash
cd src
python evaluation.py
```

**Outputs:**
- `outputs/classification_report.json` — per-class precision, recall, F1-score, and support
- `outputs/confusion_matrix.png` — heatmap (Blues colormap, 300 DPI)

---

## Inference

### Local Inference (`src/inference.py`)

Processes a video file frame-by-frame, displays the annotated output live in a window using `cv2.imshow`, and saves a JSON shot event log on exit.

```bash
cd src
python inference.py
```

**Configuration (top of `inference.py`):**

| Variable               | Default                       | Description                              |
|------------------------|-------------------------------|------------------------------------------|
| `input_video`          | `data/input_sample_video.mp4` | Path to input video                      |
| `result_json`          | `outputs/shot_results.json`   | Path to save shot event log              |
| `confidence_threshold` | `0.5`                         | Minimum GRU confidence to display label  |
| `classify_every_n`     | `3`                           | Run GRU classifier every N frames        |
| `seq_len`              | `20`                          | Rolling keypoint buffer length           |

**Controls:** Press `Q` to quit early.

---

### Google Colab Inference (`inference_colab.ipynb`)

The Colab notebook runs the **identical pipeline** headlessly — no display window — and writes every annotated frame directly to a video file using `cv2.VideoWriter`. This is the recommended approach if you don't have a local GPU.

**Runtime:** T4 GPU (`Runtime > Change runtime type > T4 GPU`)

**Steps:**

1. Open `inference_colab.ipynb` in Google Colab.
2. Upload your input video (`input_sample_video.mp4`) and model weights (`custom_gru.pth`) to the Colab `/content/` directory, or mount Google Drive and update the paths.
3. Run all cells. A `tqdm` progress bar tracks frame-level progress.

**Default paths (top of notebook):**

| Variable       | Default Path                      |
|----------------|-----------------------------------|
| `input_video`  | `/content/input_sample_video.mp4` |
| `output_video` | `/content/output.mp4`             |
| `result_json`  | `/content/shot_results.json`      |

**Outputs:**
- `/content/output.mp4` — fully annotated video with skeleton overlay, shot labels, ball/racket detections, and HUD
- `/content/shot_results.json` — JSON log of all shot events

**Observed throughput:** ~9.93 frames/sec on a T4 GPU (8125 frames processed).

**Shot event log format:**

```json
[
  {
    "player_id": 1,
    "shot_type": "forehand",
    "frame": 63
  }
]
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback supported but slow for inference)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/padel-analytics-final.git
cd padel-analytics

# 2. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-download YOLO weights (optional — auto-downloaded on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt'); YOLO('yolov8x.pt')"

# 5. Place annotation JSONs under data/annotations/
# 6. Place trained GRU weights at models/custom_gru.pth
```


> **For Google Colab**, install in the first notebook cell:
> ```python
> !pip install ultralytics opencv-python-headless torch torchvision tqdm scikit-learn
> ```

---

## Challenges Faced

### 1. Ball Detection — Small Object Size

The padel/tennis ball is extremely small in video frames, making it very difficult for general-purpose detectors to reliably detect.

- **Attempt 1 — Pretrained YOLOv8x (COCO):** Ball detection recall was poor due to the ball's small pixel footprint in full-resolution frames.
- **Attempt 2 — Custom YOLOv8 trained on a ball-only dataset:** Ball detection improved significantly, but the model had no concept of rackets since they were entirely absent from the training data.
- **Attempt 3 — Combined ball + racket dataset:** Merging two separate datasets produced inconsistent detection across both classes. The model struggled to generalise simultaneously, likely due to domain mismatch and class imbalance between the two sources.
- **Final decision:** Reverted to the pretrained `yolov8x.pt` (COCO), which natively includes both `sports ball` and `tennis racket`. While not perfect for sub-pixel ball detection, it gave the most reliable and consistent dual-object detection without additional data engineering.

> **Lesson learned:** Combining datasets from separate sources requires per-image co-annotation of all target classes, balanced sampling, and often domain-specific augmentation. A single unified dataset with both objects annotated per image would significantly improve results.

---

### 2. Keypoint Count Mismatch (17 vs 18)

YOLOv8s-Pose outputs 17 COCO keypoints, but the training dataset annotations include 18 keypoints (17 COCO + synthetic neck). To keep the GRU input consistently at 54 dimensions, a synthetic **neck keypoint** is computed at inference time as the midpoint of the left and right shoulder keypoints (COCO indices 5 and 6), exactly mirroring what was done during dataset preparation.

---

### 3. Shot Label Flicker

Since the GRU runs on a rolling 20-frame buffer and is triggered every 3 frames, a player's predicted label can oscillate between classes between consecutive runs. This was mitigated by:

- Only logging a new shot event when the predicted label **changes** from the previous one for that player.
- Applying a `confidence_threshold = 0.5` to suppress low-confidence predictions from being displayed or logged.

---

### 4. Multi-Player Tracking Instability

ByteTrack can reassign player track IDs when a player briefly leaves or is occluded in the frame. When a track ID resets, the associated rolling keypoint buffer is cleared and the player goes unclassified until the buffer fills back up to 20 frames. No additional re-identification module was added in this version.

---

Detailed per-class precision, recall, and F1-score are saved to `outputs/classification_report.json` after running `src/evaluation.py`.

---



## Future Work

- Train a more precise GRU model on a larger, padel-specific keypoint dataset with better class balance and longer sequence lengths for improved shot               classification accuracy 
- Train a dedicated **padel ball detector** using mosaic augmentation and image tiling for improved small-object detection
- Add a **re-identification module** to stabilise player track IDs across occlusions and re-entries

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — pose estimation and object detection
- [Tennis Player Actions Dataset](https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset) — Kaggle, by orvile
- [PyTorch](https://pytorch.org/) — deep learning framework
- [OpenCV](https://opencv.org/) — video processing and frame annotation
