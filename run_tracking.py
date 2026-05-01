"""
run_tracking.py
===============
Run YOLOv11 + ByteTrack tracking on PersonPath22 videos.

For each (model, video) pair this script:
  1. Calls model.track(persist=True, tracker="bytetrack.yaml") frame by frame.
  2. Keeps only person tracks (class 0).
  3. Saves per-frame tracks to  outputs/tracks/<model>/<uid>.csv
  4. Saves a MOTChallenge-format .txt to  outputs/tracks/<model>/mot/<uid>.txt
     (compatible with TrackEval / run_person_path_22.py)
  5. Optionally writes an annotated .mp4 to  outputs/videos_annotated/<model>/

MOTChallenge output format (for TrackEval)
──────────────────────────────────────────
<frame_1based>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1

Usage
-----
    python run_tracking.py
    python run_tracking.py --no_video --max_videos 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

import config as cfg
from utils.gluoncv_utils import get_split_uids, write_mot_result
from utils.video_utils import (
    AnnotatedVideoWriter,
    Timer,
    get_video_info,
    list_videos,
    open_tracking_csv,
)
from utils.yolo_utils import draw_tracks, load_model, model_variant_name, parse_tracks


# ─────────────────────────────────────────────────────────────────────────────
# Core tracking loop for one (model, video) pair
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking_on_video(
    model,
    model_name: str,
    video_path: Path,
    frame_skip: int,
    conf: float,
    imgsz: int,
    tracker_cfg: str,
    track_dir: Path,
    save_video: bool,
    annotated_dir: Path,
) -> dict:
    """Run ByteTrack tracking on one video. Returns a summary dict."""
    uid  = video_path.stem
    info = get_video_info(video_path)
    print(
        f"    {uid}  |  {info['total_frames']} frames  |  "
        f"{info['fps']:.1f} fps  |  {info['width']}×{info['height']}"
    )

    # ── output paths ──────────────────────────────────────────────────────────
    model_dir = track_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    mot_dir = model_dir / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = model_dir / f"{uid}.csv"
    mot_path = mot_dir / f"{uid}.txt"

    fh, csv_writer = open_tracking_csv(csv_path)

    # ── optional annotated video ──────────────────────────────────────────────
    video_writer = None
    if save_video:
        ann_dir = annotated_dir / model_name
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_path = ann_dir / f"{uid}_tracked.mp4"
        video_writer = AnnotatedVideoWriter(
            ann_path, info["fps"], info["width"], info["height"]
        )

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fh.close()
        raise IOError(f"Cannot open {video_path}")

    mot_rows    = []          # accumulate MOTChallenge rows
    frame_idx   = 0           # 0-based
    processed   = 0
    total_start = time.perf_counter()
    infer_timer = Timer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_id = frame_idx + 1   # 1-based (MOTChallenge convention)

            infer_timer.start()
            # persist=True keeps ByteTrack state alive across frames
            results = model.track(
                source  = frame,
                classes = [cfg.PERSON_CLASS_ID],
                conf    = conf,
                imgsz   = imgsz,
                tracker = tracker_cfg,
                persist = True,        # ← crucial for cross-frame track IDs
                verbose = False,
                stream  = False,
            )
            infer_timer.stop()

            result = results[0] if isinstance(results, list) else results
            tracks = parse_tracks(result, cfg.PERSON_CLASS_ID, conf)

            # ── CSV: one row per track ─────────────────────────────────────────
            for (tid, x1, y1, x2, y2, score, cls) in tracks:
                csv_writer.writerow([frame_idx, frame_id, tid, x1, y1, x2, y2, score, cls])

            # ── MOTChallenge rows ─────────────────────────────────────────────
            for (tid, x1, y1, x2, y2, score, cls) in tracks:
                mot_rows.append((frame_id, tid, x1, y1, x2, y2, score))

            # ── annotated frame ───────────────────────────────────────────────
            if video_writer is not None:
                ann_frame = draw_tracks(frame.copy(), tracks)
                cv2.putText(
                    ann_frame,
                    f"{uid}  frame {frame_id}  [{model_name}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                video_writer.write(ann_frame)

            processed += 1
        frame_idx += 1

    cap.release()
    fh.close()
    if video_writer is not None:
        video_writer.__exit__(None, None, None)

    # ── write MOTChallenge file ───────────────────────────────────────────────
    write_mot_result(mot_rows, mot_path)

    # Reset ByteTrack state so next video starts fresh
    try:
        model.predictor = None
    except Exception:
        pass

    total_runtime = time.perf_counter() - total_start
    infer_sec     = infer_timer.elapsed
    fps           = processed / infer_sec if infer_sec > 0 else 0.0

    print(
        f"      → {processed} frames | "
        f"infer {infer_sec:.1f}s | FPS={fps:.1f} | total {total_runtime:.1f}s"
    )

    return {
        "model_name":        model_name,
        "video_uid":         uid,
        "num_frames":        processed,
        "inference_sec":     round(infer_sec, 3),
        "total_runtime_sec": round(total_runtime, 3),
        "fps":               round(fps, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 65)
    print("YOLOv11 + ByteTrack Tracking — PersonPath22")
    print("=" * 65)

    if not args.video_dir.exists():
        sys.exit(f"[ERROR] VIDEO_DIR not found: {args.video_dir}")

    split_uids = []
    if cfg.SPLITS_PATH.exists():
        split_uids = get_split_uids(cfg.SPLITS_PATH, cfg.EVAL_SPLIT)

    videos = list_videos(args.video_dir, uid_filter=split_uids or None)
    if args.max_videos:
        videos = videos[: args.max_videos]
    print(f"Videos: {len(videos)}  (split='{cfg.EVAL_SPLIT}')\n")

    cfg.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    cfg.ANNOTATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for weights in args.models:
        variant = model_variant_name(weights)
        print(f"\n{'─'*55}")
        print(f"Model: {variant}  |  tracker: {args.tracker}")
        print(f"{'─'*55}")
        model = load_model(weights, use_gpu=cfg.USE_GPU)

        for video_path in videos:
            summary = run_tracking_on_video(
                model        = model,
                model_name   = variant,
                video_path   = video_path,
                frame_skip   = args.frame_skip,
                conf         = args.conf,
                imgsz        = args.imgsz,
                tracker_cfg  = args.tracker,
                track_dir    = cfg.TRACKING_DIR,
                save_video   = args.save_video,
                annotated_dir= cfg.ANNOTATED_VIDEO_DIR,
            )
            all_summaries.append(summary)

    # ── save tracking FPS summary ──────────────────────────────────────────────
    import csv
    fps_path = cfg.METRICS_DIR / "tracking_fps.csv"
    fps_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fps_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "video_uid", "num_frames",
            "inference_sec", "total_runtime_sec", "fps",
        ])
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"\nTracking FPS summary → {fps_path}")
    print(
        "\nMOTChallenge-format outputs saved under:\n"
        f"  {cfg.TRACKING_DIR}/<model>/mot/<uid>.txt\n"
        "These can be evaluated with TrackEval / run_person_path_22.py\n"
    )
    print("Tracking complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv11+ByteTrack on PersonPath22."
    )
    parser.add_argument("--models",     nargs="+", default=cfg.MODELS)
    parser.add_argument("--video_dir",  type=Path,  default=cfg.VIDEO_DIR)
    parser.add_argument("--max_videos", type=int,   default=cfg.MAX_VIDEOS)
    parser.add_argument("--frame_skip", type=int,   default=cfg.FRAME_SKIP)
    parser.add_argument("--conf",       type=float, default=cfg.CONFIDENCE_THRESHOLD)
    parser.add_argument("--imgsz",      type=int,   default=cfg.INFERENCE_IMGSZ)
    parser.add_argument("--tracker",    type=str,   default=cfg.TRACKER_CONFIG)
    parser.add_argument(
        "--no_video", dest="save_video", action="store_false",
        help="Skip saving annotated tracking videos (faster)",
    )
    parser.set_defaults(save_video=cfg.SAVE_TRACKING_VIDEO)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
