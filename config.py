"""
config.py
=========
Central configuration for the YOLOv11 PersonPath22 benchmark.

PersonPath22 dataset layout after running the official download.py:
--------------------------------------------------------------------
tracking-dataset/dataset/personpath22/
├── raw_data/
│   └── videos/
│       ├── uid_vid_00000.mp4
│       ├── uid_vid_00001.mp4
│       └── ...  (236 total; IDs 138 and 139 are skipped)
└── annotation/
    ├── anno_amodal_2022/          <-- FOLDER of per-video JSONs (amodal boxes)
    │   ├── uid_vid_00000.json
    │   ├── uid_vid_00001.json
    │   └── ...
    ├── anno_visible_2022/         <-- FOLDER of per-video JSONs (visible boxes)
    │   ├── uid_vid_00000.json
    │   └── ...
    └── splits.json

Per-video annotation JSON structure:
    {
      "<frame_id_str>": {          # 1-based frame number (string key)
        "labels": [
          { "box": [x1, y1, x2, y2], "label": "person", ... },
          ...
        ]
      },
      ...
    }
Only annotated key-frames (~5 FPS) appear -- most frames are absent.

Edit the paths below before running anything.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET PATHS  ← Set DATASET_ROOT before running
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path(
    r"F:\Alale\YOlo v11\personpath_yolo11\tracking-dataset\dataset\personpath22"
)

# Folder containing the uid_vid_*.mp4 files
# (the zip extracted into raw_data/videos/ not raw_data/ directly)
VIDEO_DIR = DATASET_ROOT / "raw_data" / "videos"

# ANNOTATION_PATH must point to the FOLDER, not a .json file.
# Choose one:
#   anno_amodal_2022   → full (amodal) body boxes  [recommended for detection eval]
#   anno_visible_2022  → only visible parts of each person
ANNOTATION_PATH = DATASET_ROOT / "annotation" / "anno_amodal_2022"

# Path to splits.json (train / val / test split info)
SPLITS_PATH = DATASET_ROOT / "annotation" / "splits.json"

# Which split to evaluate on: "train" | "val" | "test"
EVAL_SPLIT = "test"

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]

PERSON_CLASS_ID      = 0      # COCO class id for "person"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD        = 0.45
INFERENCE_IMGSZ      = 640    # square input size; try 320 or 480 for speed tests

# ─────────────────────────────────────────────────────────────────────────────
# 3. FRAME-SKIPPING EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

# Process every Nth frame.  1 = every frame, 2 = every other, 5 = every 5th.
# PersonPath22 GT is annotated at ~5 FPS, so mAP is computed only on
# annotated frames regardless of this setting.
FRAME_SKIP = 1

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRACKING
# ─────────────────────────────────────────────────────────────────────────────

TRACKER_CONFIG      = "bytetrack.yaml"
SAVE_TRACKING_VIDEO = True

# ─────────────────────────────────────────────────────────────────────────────
# 5. OUTPUT PATHS  (auto-created at runtime)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_ROOT         = Path("outputs")
DETECTION_DIR       = OUTPUT_ROOT / "detections"
TRACKING_DIR        = OUTPUT_ROOT / "tracks"
METRICS_DIR         = OUTPUT_ROOT / "metrics"
ANNOTATED_VIDEO_DIR = OUTPUT_ROOT / "videos_annotated"

# ─────────────────────────────────────────────────────────────────────────────
# 6. MISC
# ─────────────────────────────────────────────────────────────────────────────

USE_GPU    = True    # set False to force CPU
MAX_VIDEOS = None    # set to e.g. 3 for a quick smoke-test
RANDOM_SEED = 42
