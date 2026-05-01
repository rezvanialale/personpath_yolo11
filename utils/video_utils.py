"""
utils/video_utils.py
====================
Video I/O helpers: listing files, iterating frames, writing annotated output,
timing, and CSV writing.
"""

import csv
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Video metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_video_info(video_path: Path) -> dict:
    """
    Return basic metadata for a video file.

    Returns dict with keys:
        path, video_uid, width, height, fps, total_frames
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "path":         video_path,
        "video_uid":    video_path.stem,          # e.g. "uid_vid_00000"
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def list_videos(
    video_dir: Path,
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    uid_filter: Optional[List[str]] = None,
) -> List[Path]:
    """
    Return a sorted list of video files in *video_dir*.

    Parameters
    ----------
    uid_filter : if given, only return videos whose stem is in this list.
                 Used to restrict to a specific dataset split.
    """
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    if uid_filter:
        uid_set = set(uid_filter)
        videos = [v for v in videos if v.stem in uid_set]
    return videos


# ─────────────────────────────────────────────────────────────────────────────
# Annotated video writer
# ─────────────────────────────────────────────────────────────────────────────

class AnnotatedVideoWriter:
    """
    Context-manager that writes BGR frames to a .mp4 file.

    Usage
    -----
    with AnnotatedVideoWriter(out_path, fps, width, height) as writer:
        writer.write(frame)
    """

    def __init__(self, out_path: Path, fps: float, width: int, height: int):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Cannot open VideoWriter for {out_path}")

    def write(self, frame):
        self._writer.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

# Per-frame detection output columns
DETECTION_CSV_HEADER = [
    "frame_idx",       # 0-based OpenCV index
    "frame_id",        # 1-based (matches PersonPath22 annotation convention)
    "x1", "y1", "x2", "y2",
    "confidence", "class_id", "model_name",
]

# Per-frame tracking output columns
TRACKING_CSV_HEADER = [
    "frame_idx",       # 0-based
    "frame_id",        # 1-based (MOTChallenge convention)
    "track_id",
    "x1", "y1", "x2", "y2",
    "confidence", "class_id",
]


def open_detection_csv(out_path: Path):
    """Open a detection CSV for writing. Returns (file_handle, csv.writer)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(DETECTION_CSV_HEADER)
    return fh, writer


def open_tracking_csv(out_path: Path):
    """Open a tracking CSV for writing. Returns (file_handle, csv.writer)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(TRACKING_CSV_HEADER)
    return fh, writer


# ─────────────────────────────────────────────────────────────────────────────
# Simple wall-clock timer
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Accumulating wall-clock timer.  Can be started/stopped multiple times."""

    def __init__(self):
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer not started.")
        self.elapsed += time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def reset(self):
        self._start = None
        self.elapsed = 0.0

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
