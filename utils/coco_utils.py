"""
utils/coco_utils.py
===================
Helpers for working with COCO-format annotations and for converting
PersonPath22 MOT-style annotations into COCO JSON format.

PersonPath22 annotation notes
------------------------------
PersonPath22 can ship annotations in two formats:

  (A) COCO JSON  —  a single JSON file with 'images', 'annotations', 'categories'.
      Set ANNOTATION_FORMAT = "coco" in config.py.

  (B) MOT-style  —  one text file per video sequence, each line:
          frame_id, track_id, x, y, w, h, conf, class, visibility
      Set ANNOTATION_FORMAT = "mot" in config.py.

If your annotations are in format (B), call `convert_mot_folder_to_coco()` once
(or run  `python utils/coco_utils.py`)  to produce the COCO JSON that
evaluate_map.py expects.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────
# COCO JSON loading
# ──────────────────────────────────────────────────────────────

def load_coco_gt(annotation_path: Path) -> dict:
    """
    Load a COCO-format ground-truth JSON file.

    Returns the parsed dict with keys: 'images', 'annotations', 'categories'.
    """
    with open(annotation_path, "r") as f:
        gt = json.load(f)
    print(f"  GT loaded: {len(gt['images'])} images, {len(gt['annotations'])} annotations")
    return gt


def build_image_id_map(gt: dict) -> Dict[str, int]:
    """
    Return a dict mapping  file_name → image_id  from a COCO GT dict.
    Used to align predicted image_ids with ground-truth image_ids.
    """
    return {img["file_name"]: img["id"] for img in gt["images"]}


# ──────────────────────────────────────────────────────────────
# MOT → COCO conversion
# ──────────────────────────────────────────────────────────────

def convert_mot_folder_to_coco(
    mot_dir: Path,
    video_dir: Path,
    out_json: Path,
    person_category_id: int = 1,   # COCO uses 1-indexed categories
    visibility_threshold: float = 0.0,
) -> dict:
    """
    Convert a folder of MOT-style annotation .txt files to a single COCO JSON.

    MOT format per line:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>,
        <conf>, <class>, <visibility>

    Parameters
    ----------
    mot_dir              : folder containing one .txt per video
    video_dir            : folder containing the corresponding video files
                           (used only to determine frame width/height)
    out_json             : where to write the output COCO JSON
    person_category_id   : category id used in the output (default 1 for COCO)
    visibility_threshold : skip annotations with visibility below this

    Returns
    -------
    The COCO-format dict (also written to out_json).
    """
    import cv2

    coco: dict = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": person_category_id, "name": "person", "supercategory": "person"}],
    }

    image_id    = 0
    ann_id      = 0

    mot_files = sorted(mot_dir.glob("*.txt"))
    if not mot_files:
        raise FileNotFoundError(f"No .txt files found in {mot_dir}")

    for mot_file in mot_files:
        video_id = mot_file.stem   # e.g. "video_001"

        # Try to find the matching video file to get resolution
        video_path = _find_video(video_dir, video_id)
        if video_path is not None:
            cap = cv2.VideoCapture(str(video_path))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            print(f"  [WARNING] Could not find video for {video_id}; using width=0, height=0")
            width, height = 0, 0

        # Parse MOT file
        frame_set: Dict[int, int] = {}   # frame_number → image_id

        with open(mot_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                frame_num = int(parts[0])
                # track_id = int(parts[1])   # not needed for detection GT
                bb_left  = float(parts[2])
                bb_top   = float(parts[3])
                bb_w     = float(parts[4])
                bb_h     = float(parts[5])
                conf     = float(parts[6]) if len(parts) > 6 else 1.0
                # class  = int(parts[7])   # 1 = pedestrian in MOT
                visibility = float(parts[8]) if len(parts) > 8 else 1.0

                if visibility < visibility_threshold:
                    continue

                # Register image entry for this frame if not seen yet
                if frame_num not in frame_set:
                    image_id += 1
                    frame_set[frame_num] = image_id
                    coco["images"].append({
                        "id":        image_id,
                        "file_name": f"{video_id}/frame_{frame_num:06d}.jpg",
                        "width":     width,
                        "height":    height,
                        "video_id":  video_id,
                        "frame_id":  frame_num,
                    })

                # Add annotation
                ann_id += 1
                coco["annotations"].append({
                    "id":          ann_id,
                    "image_id":    frame_set[frame_num],
                    "category_id": person_category_id,
                    "bbox":        [bb_left, bb_top, bb_w, bb_h],  # COCO: [x, y, w, h]
                    "area":        bb_w * bb_h,
                    "iscrowd":     0,
                    "score":       conf,
                })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"  MOT → COCO conversion done: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations → {out_json}")
    return coco


def _find_video(video_dir: Path, stem: str) -> Optional[Path]:
    """Find a video file whose stem matches *stem* (any common extension)."""
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = video_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────
# Building a COCO-style predictions list
# ──────────────────────────────────────────────────────────────

def build_coco_predictions(
    detections_per_image: List[dict],
    person_category_id: int = 1,
) -> List[dict]:
    """
    Convert a list of detection dicts to COCO-predictions format for pycocotools.

    Each dict in detections_per_image should have:
        image_id, x1, y1, x2, y2, score

    Returns a list of dicts with keys: image_id, category_id, bbox, score.
    bbox is COCO-style [x, y, width, height].
    """
    preds = []
    for det in detections_per_image:
        x1, y1 = det["x1"], det["y1"]
        x2, y2 = det["x2"], det["y2"]
        preds.append({
            "image_id":    det["image_id"],
            "category_id": person_category_id,
            "bbox":        [x1, y1, x2 - x1, y2 - y1],
            "score":       det["score"],
        })
    return preds


# ──────────────────────────────────────────────────────────────
# CLI entry point for MOT → COCO conversion
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as cfg

    if cfg.ANNOTATION_FORMAT == "mot":
        out = cfg.ANNOTATION_PATH  # write the converted JSON here
        convert_mot_folder_to_coco(
            mot_dir   = cfg.MOT_ANNOTATION_DIR,
            video_dir = cfg.VIDEO_DIR,
            out_json  = out,
        )
        print(f"Saved COCO JSON to {out}")
    else:
        print("ANNOTATION_FORMAT is already 'coco'; nothing to convert.")
