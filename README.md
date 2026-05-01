# YOLOv11 Benchmark on PersonPath22

A modular Python pipeline that benchmarks **YOLOv11n / YOLOv11s / YOLOv11m** for
person detection and tracking on the real-world
[PersonPath22](https://amazon-science.github.io/tracking-dataset/personpath22.html)
video dataset.

This is one component of a group project comparing multiple detectors
(**YOLOv8, RT-DETR, Faster R-CNN, YOLOv11**) on PersonPath22.
See [Comparison with Other Models](#comparison-with-other-models) below.

---

## Dataset overview

PersonPath22 is a large-scale multi-person tracking dataset introduced at ECCV 2022.

| Property | Value |
|---|---|
| Videos | 236 `.mp4` files (`uid_vid_00000` – `uid_vid_00236`, IDs 138/139 skipped) |
| Annotation format | **GluonCV motion-dataset JSON** (`anno_visible_2022.json` / `anno_amodal_2022.json`) |
| Annotation rate | Key-frames at **~5 FPS** (full-FPS annotations listed as "To Appear") |
| Bbox types | **Visible** (partially visible bodies) and **Amodal** (full estimated body) |
| Official eval | [TrackEval](https://github.com/JonathonLuiten/TrackEval) with MOTChallenge format |
| Splits | `splits.json` provides train / val / test UIDs |

### GluonCV annotation structure

```json
{
  "database": {
    "uid_vid_00000": {
      "nframes": 900,
      "fps": 30.0,
      "resolution": [720, 1280],
      "annotations": {
        "1": { "bboxes": { "1": [x,y,w,h], "6": [x,y,w,h], ... } },
        "2": { "bboxes": { "1": [x,y,w,h], ... } }
      }
    },
    ...
  }
}
```

- Frame IDs in `bboxes` are **1-based** integers (as strings).
- Only annotated key-frames appear — unannotated frames are simply absent.
- Bounding boxes are `[x, y, width, height]` in pixel coordinates.

---

## Project structure

```
personpath_yolo11/
├── config.py                ← Edit the three dataset paths HERE first
├── run_detection.py         ← Step 2: detect persons on all videos
├── run_tracking.py          ← Step 3: ByteTrack tracking
├── evaluate_map.py          ← Step 4: mAP50-95 / mAP50 / mAP75
├── summarize_results.py     ← Step 5: build final summary CSVs
├── requirements.txt
├── README.md
└── utils/
    ├── gluoncv_utils.py     ← Parse PersonPath22 annotations; convert to COCO
    ├── video_utils.py       ← Video I/O, CSV writers, timer
    └── yolo_utils.py        ← Model loading, result parsing, drawing
```

### Output layout (auto-created)

```
outputs/
├── detections/
│   ├── yolo11n/
│   │   ├── uid_vid_00000.csv          ← frame-level bboxes
│   │   ├── uid_vid_00000_preds.json   ← COCO-format predictions
│   │   └── ...
│   ├── yolo11s/
│   └── yolo11m/
├── tracks/
│   ├── yolo11n/
│   │   ├── uid_vid_00000.csv          ← frame-level track IDs
│   │   └── mot/
│   │       └── uid_vid_00000.txt      ← MOTChallenge format (TrackEval input)
│   ├── yolo11s/
│   └── yolo11m/
├── metrics/
│   ├── gt_coco_test.json              ← converted GT (auto-generated)
│   ├── detection_fps.csv
│   ├── tracking_fps.csv
│   ├── map_results.csv
│   ├── per_video_summary.csv
│   └── summary.csv                    ← FINAL TABLE
└── videos_annotated/
    ├── yolo11n/
    ├── yolo11s/
    └── yolo11m/
```

---

## Quick-start (step by step)

### Step 0 — Requirements

- Python 3.9 or later
- A CUDA GPU is **strongly recommended** (36 × 30-min videos at 30 FPS is a lot of frames)
- PersonPath22 downloaded using the official script (see Step 1)

---

### Step 1 — Download PersonPath22

```bash
# Clone the official Amazon dataset repo
git clone https://github.com/amazon-science/tracking-dataset.git
cd tracking-dataset

# Install the download script dependencies
pip install awscli requests tqdm

# Run the downloader (~200 GB; takes hours; needs ffmpeg in PATH)
python download.py
```

After completion you will have:

```
tracking-dataset/dataset/personpath22/
├── raw_data/
│   ├── uid_vid_00000.mp4
│   ...
└── annotation/
    ├── anno_visible_2022.json
    ├── anno_amodal_2022.json
    └── splits.json
```

---

### Step 2 — Install benchmark dependencies

```bash
cd personpath_yolo11        # this project folder

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt

# GPU users — install matching PyTorch+CUDA first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Windows + pycocotools**: If compilation fails, use `pip install pycocotools-windows`

---

### Step 3 — Configure paths

Open **`config.py`** and set `DATASET_ROOT` to wherever you cloned the Amazon repo:

```python
DATASET_ROOT = Path("/path/to/tracking-dataset/dataset/personpath22")
```

`VIDEO_DIR` and `ANNOTATION_PATH` are derived from `DATASET_ROOT` automatically.

**Optional settings in config.py:**

| Setting | Default | Effect |
|---|---|---|
| `ANNOTATION_FILE` | `"anno_amodal_2022.json"` | Use visible or amodal GT boxes |
| `EVAL_SPLIT` | `"test"` | Which PersonPath22 split to use |
| `FRAME_SKIP` | `1` | Process every Nth frame |
| `INFERENCE_IMGSZ` | `640` | Inference resolution (try 320 or 480) |
| `MAX_VIDEOS` | `None` | Limit to N videos for quick tests |
| `SAVE_TRACKING_VIDEO` | `True` | Write annotated .mp4 files |
| `USE_GPU` | `True` | Set False to force CPU |

---

### Step 4 — Run detection

```bash
python run_detection.py
```

This loops over all test-split videos × 3 models. YOLOv11 weights are
downloaded automatically from Ultralytics on first run (~50–170 MB each).

**Quick smoke-test** (3 videos, every 5th frame):
```bash
python run_detection.py --max_videos 3 --frame_skip 5
```

---

### Step 5 — Run tracking

```bash
python run_tracking.py
```

Uses `model.track(persist=True, tracker="bytetrack.yaml")`.
`persist=True` keeps ByteTrack state alive across consecutive frames — this is
the flag that makes it a real tracker rather than per-frame detection.

Skip annotated videos (faster):
```bash
python run_tracking.py --no_video
```

MOTChallenge-format outputs are saved to `outputs/tracks/<model>/mot/<uid>.txt`
and can be evaluated with TrackEval directly (see [TrackEval section](#trackeval-tracking-metrics) below).

---

### Step 6 — Evaluate mAP

```bash
python evaluate_map.py
```

- Converts `anno_amodal_2022.json` → `outputs/metrics/gt_coco_test.json` (once, cached).
- Matches predictions to GT by `image_id = video_idx × 1_000_000 + frame_id_1based`.
- Frames with no GT annotation are automatically ignored by pycocotools (correct behaviour —
  PersonPath22 annotates only key-frames at ~5 FPS).

---

### Step 7 — Build summary table

```bash
python summarize_results.py
```

Produces `outputs/metrics/summary.csv` and prints a comparison table.

---

### Run all steps at once

```bash
python run_detection.py && \
python run_tracking.py --no_video && \
python evaluate_map.py && \
python summarize_results.py
```

---

## TrackEval — tracking metrics

For MOTA, MOTP, HOTA (official PersonPath22 tracking evaluation):

```bash
# Clone TrackEval
git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval
pip install -r requirements.txt

# Download the official PersonPath22 GT data package (~3 MB)
# from: https://github.com/JonathonLuiten/TrackEval  (person_path_22_data.zip)
# Extract into TrackEval/data/

# Run evaluation for one model (e.g. yolo11n)
python scripts/run_person_path_22.py \
    --TRACKERS_FOLDER /path/to/personpath_yolo11/outputs/tracks \
    --TRACKERS_TO_EVAL yolo11n \
    --GT_FOLDER TrackEval/data/gt/person_path_22
```

The MOTChallenge `.txt` files in `outputs/tracks/<model>/mot/` match the
format TrackEval expects: `<frame>,<id>,<left>,<top>,<w>,<h>,<conf>,-1,-1,-1`.

---

## Summary CSV columns

### `summary.csv` (one row per model)

| Column | Description |
|---|---|
| `model_name` | `yolo11n`, `yolo11s`, `yolo11m` |
| `num_videos` | Number of test-split videos processed |
| `total_frames` | Total frames passed through inference |
| `avg_fps` | Mean inference-only FPS across videos |
| `total_detections` | Total person detections saved |
| `mAP50_95` | COCO primary metric (IoU 0.50:0.95) |
| `mAP50` | AP at IoU = 0.50 |
| `mAP75` | AP at IoU = 0.75 |

### `per_video_summary.csv` (one row per model × video)

Same columns plus `video_uid`, `num_frames`, `fps` per video.

---

## Comparison with other models

All benchmarks in the group project share the same evaluation protocol:

| Model family | Variants | Script/Repo |
|---|---|---|
| **YOLOv11** (this repo) | yolo11n / yolo11s / yolo11m | — |
| YOLOv8 | yolov8n / yolov8s / yolov8m | teammate 1 |
| RT-DETR | RT-DETR-L / RT-DETR-X | teammate 2 |
| Faster R-CNN | ResNet-50 / ResNet-101 | teammate 3 |

**How to combine results for the final report:**

```python
import pandas as pd

dfs = [
    pd.read_csv("yolo11/outputs/metrics/summary.csv"),
    pd.read_csv("yolov8/outputs/metrics/summary.csv"),
    pd.read_csv("rtdetr/outputs/metrics/summary.csv"),
    pd.read_csv("frcnn/outputs/metrics/summary.csv"),
]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("combined_benchmark.csv", index=False)
print(combined[["model_name", "avg_fps", "mAP50_95", "mAP50", "mAP75"]].to_string())
```

**Key comparison axes:**

1. **Accuracy** — mAP50-95, mAP50, mAP75 on the identical PersonPath22 test split
2. **Speed** — avg_fps at resolution 640, frame_skip 1
3. **Speed–accuracy curve** — scatter plot mAP50 vs FPS across all variants
4. **Scale sensitivity** — re-run with `INFERENCE_IMGSZ = 320` and `480`
5. **Tracking quality** — HOTA, MOTA, IDF1 from TrackEval

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `pycocotools` compile fails on Windows | `pip install pycocotools-windows` |
| CUDA out of memory | Set `INFERENCE_IMGSZ=320` in config.py |
| Videos not found | Check `DATASET_ROOT` in config.py |
| `model.track() AttributeError: id` | `pip install -U ultralytics` |
| mAP is 0.0 for all models | Check that `EVAL_SPLIT` matches the split in `splits.json` |
| No GT frames match predictions | Annotations are ~5 FPS; this is expected — AP is computed on those frames only |

---

## Citation

```bibtex
@inproceedings{personpath22,
  title={Large Scale Real-world Multi-Person Tracking},
  author={Shuai, Bing and Bergamo, Alessandro and Buechler, Uta and
          Berneshawi, Andrew and Boden, Alyssa and Tighe, Joseph},
  booktitle={ECCV},
  year={2022},
  organization={Springer}
}
```
