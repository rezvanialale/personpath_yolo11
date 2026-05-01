"""
run_detection.py
================
Run YOLOv11n / YOLOv11s / YOLOv11m person detection on PersonPath22 videos.

For each (model, video) pair this script:
  1. Opens the video and iterates frames (honouring FRAME_SKIP).
  2. Runs YOLOv11 inference; keeps only person detections (class 0, conf ≥ 0.25).
  3. Saves detections to  outputs/detections/<model>/<uid>.csv
  4. Saves a COCO-style prediction JSON to  outputs/detections/<model>/<uid>_preds.json
     (image_id encodes video_index * 1_000_000 + frame_id_1based,
      matching the scheme used in utils/gluoncv_utils.py::gluoncv_to_coco)
  5. Records per-video FPS (pure inference time, excluding decode overhead).

Usage
-----
    python run_detection.py                    # run all models on all test videos
    python run_detection.py --max_videos 3     # quick smoke-test
    python run_detection.py --models yolo11n.pt --frame_skip 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

import config as cfg
from utils.gluoncv_utils import (
    build_uid_to_video_idx,
    get_split_uids,
    load_gluoncv_anno,
)
from utils.video_utils import (
    AnnotatedVideoWriter,
    Timer,
    get_video_info,
    list_videos,
    open_detection_csv,
)
from utils.yolo_utils import (
    draw_detections,
    load_model,
    model_variant_name,
    parse_detections,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core detection loop for one (model, video) pair
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_on_video(
    model,
    model_name: str,
    video_path: Path,
    video_idx: int,          # position of this UID in the uid_list — used to build image_id
    frame_skip: int,
    conf: float,
    imgsz: int,
    det_dir: Path,
) -> dict:
    """
    Run detection on a single video.  Returns a summary dict.
    """
    uid  = video_path.stem   # e.g. "uid_vid_00000"
    info = get_video_info(video_path)
    print(
        f"    {uid}  |  {info['total_frames']} frames  |  "
        f"{info['fps']:.1f} fps  |  {info['width']}×{info['height']}"
    )

    # ── output paths ──────────────────────────────────────────────────────────
    out_dir = det_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{uid}.csv"
    json_path = out_dir / f"{uid}_preds.json"

    fh, csv_writer = open_detection_csv(csv_path)

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fh.close()
        raise IOError(f"Cannot open {video_path}")

    coco_preds  = []
    frame_idx   = 0          # 0-based OpenCV counter
    processed   = 0
    total_start = time.perf_counter()
    infer_timer = Timer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_id = frame_idx + 1   # 1-based, matches annotation convention

            # ── INFERENCE (timed) ─────────────────────────────────────────────
            infer_timer.start()
            results = model.predict(
                source  = frame,
                classes = [cfg.PERSON_CLASS_ID],
                conf    = conf,
                imgsz   = imgsz,
                verbose = False,
                stream  = False,
            )
            infer_timer.stop()

            result     = results[0] if isinstance(results, list) else results
            detections = parse_detections(result, cfg.PERSON_CLASS_ID, conf)

            # ── image_id: same scheme as gluoncv_to_coco() ───────────────────
            # video_idx * 1_000_000 + frame_id
            image_id = video_idx * 1_000_000 + frame_id

            # ── write CSV (one row per detection) ─────────────────────────────
            for (x1, y1, x2, y2, score, cls) in detections:
                csv_writer.writerow([
                    frame_idx, frame_id,
                    x1, y1, x2, y2,
                    score, cls, model_name,
                ])
                # COCO prediction entry: bbox is [x, y, w, h]
                coco_preds.append({
                    "image_id":    image_id,
                    "category_id": 1,               # COCO person = 1
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],
                    "score":       score,
                })

            processed += 1
        frame_idx += 1

    cap.release()
    fh.close()

    # ── save COCO prediction JSON ─────────────────────────────────────────────
    with open(json_path, "w") as jf:
        json.dump(coco_preds, jf)

    total_runtime = time.perf_counter() - total_start
    infer_sec     = infer_timer.elapsed
    fps           = processed / infer_sec if infer_sec > 0 else 0.0

    print(
        f"      → {processed} frames | "
        f"infer {infer_sec:.1f}s | FPS={fps:.1f} | total {total_runtime:.1f}s | "
        f"{len(coco_preds)} detections"
    )

    return {
        "model_name":        model_name,
        "video_uid":         uid,
        "video_idx":         video_idx,
        "num_frames":        processed,
        "inference_sec":     round(infer_sec, 3),
        "total_runtime_sec": round(total_runtime, 3),
        "fps":               round(fps, 2),
        "num_detections":    len(coco_preds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 65)
    print("YOLOv11 Detection — PersonPath22")
    print("=" * 65)

    if not args.video_dir.exists():
        sys.exit(
            f"[ERROR] VIDEO_DIR not found: {args.video_dir}\n"
            "Set DATASET_ROOT in config.py and re-run."
        )

    # ── load annotations to get the split uid list ────────────────────────────
    split_uids = []
    if cfg.SPLITS_PATH.exists() and cfg.ANNOTATION_PATH.exists():
        split_uids = get_split_uids(cfg.SPLITS_PATH, cfg.EVAL_SPLIT)

    # ── list videos restricted to the split ──────────────────────────────────
    videos = list_videos(args.video_dir, uid_filter=split_uids or None)
    if args.max_videos:
        videos = videos[: args.max_videos]

    # Build uid → video_idx map  (must match gluoncv_to_coco indexing)
    uid_list  = [v.stem for v in videos]
    uid_to_idx = build_uid_to_video_idx(uid_list)

    print(f"Videos to process: {len(videos)}  (split='{cfg.EVAL_SPLIT}')\n")

    cfg.DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for weights in args.models:
        variant = model_variant_name(weights)
        print(f"\n{'─'*55}")
        print(f"Model: {variant}")
        print(f"{'─'*55}")
        model = load_model(weights, use_gpu=cfg.USE_GPU)

        for video_path in videos:
            uid       = video_path.stem
            video_idx = uid_to_idx[uid]
            summary   = run_detection_on_video(
                model      = model,
                model_name = variant,
                video_path = video_path,
                video_idx  = video_idx,
                frame_skip = args.frame_skip,
                conf       = args.conf,
                imgsz      = args.imgsz,
                det_dir    = cfg.DETECTION_DIR,
            )
            all_summaries.append(summary)

    # ── save FPS / runtime summary CSV ────────────────────────────────────────
    import csv
    fps_path = cfg.METRICS_DIR / "detection_fps.csv"
    with open(fps_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "video_uid", "video_idx", "num_frames",
            "inference_sec", "total_runtime_sec", "fps", "num_detections",
        ])
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"\nDetection FPS summary → {fps_path}")
    print("Detection complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 person detection on PersonPath22."
    )
    parser.add_argument("--models",     nargs="+", default=cfg.MODELS)
    parser.add_argument("--video_dir",  type=Path,  default=cfg.VIDEO_DIR)
    parser.add_argument("--max_videos", type=int,   default=cfg.MAX_VIDEOS)
    parser.add_argument("--frame_skip", type=int,   default=cfg.FRAME_SKIP)
    parser.add_argument("--conf",       type=float, default=cfg.CONFIDENCE_THRESHOLD)
    parser.add_argument("--imgsz",      type=int,   default=cfg.INFERENCE_IMGSZ)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
