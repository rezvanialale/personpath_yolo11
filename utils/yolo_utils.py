"""
utils/yolo_utils.py
===================
Helpers for loading YOLOv11 models and parsing Ultralytics result objects.
"""

from pathlib import Path
from typing import List, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights: str, use_gpu: bool = True):
    """
    Load a YOLOv11 model via Ultralytics.

    Parameters
    ----------
    weights : str  e.g. "yolo11n.pt" — Ultralytics downloads weights on first run.
    use_gpu : bool  Use CUDA when available.

    Returns
    -------
    ultralytics.YOLO instance
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model.to(device)
    print(f"  Loaded {weights}  →  device={device}")
    return model


def model_variant_name(weights: str) -> str:
    """Return a clean variant label.  e.g. 'yolo11n.pt' → 'yolo11n'."""
    return Path(weights).stem


# ─────────────────────────────────────────────────────────────────────────────
# Parsing detection results
# ─────────────────────────────────────────────────────────────────────────────

def parse_detections(
    results,
    person_class_id: int = 0,
    conf_threshold: float = 0.25,
) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Extract person detections from an Ultralytics Results object (single frame).

    Returns
    -------
    List of (x1, y1, x2, y2, confidence, class_id)  — xyxy pixel coordinates.
    """
    detections = []
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return detections

    xyxy    = results.boxes.xyxy.cpu().numpy()
    confs   = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        cls = int(cls)
        conf = float(conf)
        if cls == person_class_id and conf >= conf_threshold:
            detections.append((float(x1), float(y1), float(x2), float(y2), conf, cls))

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Parsing tracking results
# ─────────────────────────────────────────────────────────────────────────────

def parse_tracks(
    results,
    person_class_id: int = 0,
    conf_threshold: float = 0.25,
) -> List[Tuple[int, float, float, float, float, float, int]]:
    """
    Extract tracked persons from an Ultralytics Results object (tracking mode).

    Returns
    -------
    List of (track_id, x1, y1, x2, y2, confidence, class_id)
    track_id is -1 when no ID is assigned.
    """
    tracks = []
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return tracks

    xyxy    = results.boxes.xyxy.cpu().numpy()
    confs   = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    ids     = results.boxes.id
    id_arr  = ids.cpu().numpy() if ids is not None else [-1] * len(confs)

    for (x1, y1, x2, y2), conf, cls, tid in zip(xyxy, confs, classes, id_arr):
        cls  = int(cls)
        conf = float(conf)
        tid  = int(tid) if tid is not None else -1
        if cls == person_class_id and conf >= conf_threshold:
            tracks.append((tid, float(x1), float(y1), float(x2), float(y2), conf, cls))

    return tracks


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on a BGR frame in-place."""
    import cv2
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_tracks(frame, tracks, thickness=2):
    """Draw tracked bounding boxes with track IDs on a BGR frame in-place."""
    import cv2
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 255),
        (255, 0, 128), (0, 128, 255), (255, 255, 0),
    ]
    for (tid, x1, y1, x2, y2, conf, cls) in tracks:
        color = palette[tid % len(palette)] if tid >= 0 else (200, 200, 200)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"ID:{tid} {conf:.2f}" if tid >= 0 else f"{conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame
