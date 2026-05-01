"""
summarize_results.py
====================
Merge detection FPS data and mAP results into two summary CSV files:

  outputs/metrics/per_video_summary.csv   — one row per (model, video)
  outputs/metrics/summary.csv             — one row per model (aggregated)

Then prints a formatted comparison table to the terminal.

Run this after run_detection.py and evaluate_map.py.

Usage
-----
    python summarize_results.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from utils.yolo_utils import model_variant_name


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list:
    if not path.exists():
        print(f"  [WARNING] File not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def detection_count(model_name: str, video_uid: str) -> int:
    """Count detection rows in the per-video CSV (header excluded)."""
    p = cfg.DETECTION_DIR / model_name / f"{video_uid}.csv"
    if not p.exists():
        return 0
    with open(p) as f:
        return max(0, sum(1 for _ in f) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Build per-video table
# ─────────────────────────────────────────────────────────────────────────────

PV_FIELDS = [
    "model_name", "variant", "video_uid",
    "num_frames", "total_runtime_sec", "fps",
    "num_detections",
    "mAP50_95", "mAP50", "mAP75",
]


def build_per_video(fps_rows: list, map_lookup: dict) -> list:
    rows = []
    for r in fps_rows:
        model = r.get("model_name", "")
        uid   = r.get("video_uid", "")
        mrow  = map_lookup.get(model, {})
        rows.append({
            "model_name":        model,
            "variant":           model,
            "video_uid":         uid,
            "num_frames":        r.get("num_frames", ""),
            "total_runtime_sec": r.get("total_runtime_sec", ""),
            "fps":               r.get("fps", ""),
            "num_detections":    detection_count(model, uid),
            "mAP50_95":          mrow.get("mAP50_95", ""),
            "mAP50":             mrow.get("mAP50", ""),
            "mAP75":             mrow.get("mAP75", ""),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build overall model-level summary
# ─────────────────────────────────────────────────────────────────────────────

SUM_FIELDS = [
    "model_name", "variant",
    "num_videos", "total_frames", "total_runtime_sec", "avg_fps",
    "total_detections",
    "mAP50_95", "mAP50", "mAP75",
]


def build_overall(per_video: list, map_lookup: dict) -> list:
    groups = defaultdict(list)
    for r in per_video:
        groups[r["model_name"]].append(r)

    summaries = []
    for weights in cfg.MODELS:
        model = model_variant_name(weights)
        rows  = groups.get(model, [])
        mrow  = map_lookup.get(model, {})

        if not rows:
            summaries.append({
                "model_name": model, "variant": model,
                "num_videos": 0, "total_frames": 0,
                "total_runtime_sec": 0, "avg_fps": "",
                "total_detections": 0,
                "mAP50_95": mrow.get("mAP50_95", ""),
                "mAP50":    mrow.get("mAP50", ""),
                "mAP75":    mrow.get("mAP75", ""),
            })
            continue

        fps_vals = [_f(r["fps"]) for r in rows if _f(r["fps"]) is not None]
        summaries.append({
            "model_name":        model,
            "variant":           model,
            "num_videos":        len(rows),
            "total_frames":      sum(_i(r["num_frames"]) for r in rows),
            "total_runtime_sec": round(sum(_f(r["total_runtime_sec"], 0) for r in rows), 2),
            "avg_fps":           round(sum(fps_vals) / len(fps_vals), 2) if fps_vals else "",
            "total_detections":  sum(_i(r["num_detections"]) for r in rows),
            "mAP50_95":          mrow.get("mAP50_95", ""),
            "mAP50":             mrow.get("mAP50", ""),
            "mAP75":             mrow.get("mAP75", ""),
        })
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows: list, fields: list, title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")
    if not rows:
        print("  (no data)")
        return
    widths = {f: max(len(f), max(len(str(r.get(f, ""))) for r in rows)) for f in fields}
    sep  = "  " + "  ".join("-" * widths[f] for f in fields)
    head = "  " + "  ".join(f.ljust(widths[f]) for f in fields)
    print(head)
    print(sep)
    for r in rows:
        print("  " + "  ".join(str(r.get(f, "")).ljust(widths[f]) for f in fields))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("YOLOv11 Benchmark — Results Summary")
    print("=" * 65)

    # ── load raw data ──────────────────────────────────────────────────────────
    fps_path = cfg.METRICS_DIR / "detection_fps.csv"
    if not fps_path.exists():
        fps_path = cfg.METRICS_DIR / "tracking_fps.csv"   # fallback

    fps_rows  = load_csv(fps_path)
    map_rows  = load_csv(cfg.METRICS_DIR / "map_results.csv")
    map_lookup = {r["model_name"]: r for r in map_rows}

    if not fps_rows and not map_rows:
        sys.exit(
            "[ERROR] No results found.\n"
            "Run run_detection.py and evaluate_map.py first."
        )

    # ── build tables ──────────────────────────────────────────────────────────
    per_video = build_per_video(fps_rows, map_lookup)
    overall   = build_overall(per_video, map_lookup)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pv_path = cfg.METRICS_DIR / "per_video_summary.csv"
    with open(pv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PV_FIELDS)
        writer.writeheader()
        writer.writerows(per_video)
    print(f"Per-video summary → {pv_path}")

    sum_path = cfg.METRICS_DIR / "summary.csv"
    with open(sum_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUM_FIELDS)
        writer.writeheader()
        writer.writerows(overall)
    print(f"Overall summary   → {sum_path}")

    # ── print ─────────────────────────────────────────────────────────────────
    ov_cols = ["model_name", "num_videos", "total_frames",
               "avg_fps", "total_detections", "mAP50_95", "mAP50", "mAP75"]
    print_table(overall, ov_cols, "Overall Model Comparison  (YOLOv11n / YOLOv11s / YOLOv11m)")

    pv_cols = ["model_name", "video_uid", "num_frames",
               "fps", "num_detections", "mAP50_95", "mAP50"]
    print_table(per_video, pv_cols, "Per-Video Detail")

    print("\n" + "=" * 65)
    print("YOLOv11 benchmark completed. Check outputs/metrics/summary.csv.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
